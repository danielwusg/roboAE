"""Launch a vLLM OpenAI-compatible server as the upstream for the language tool (W2).

This is the ``--vllm`` path: instead of the one-request-at-a-time ``transformers``
``qwen-language`` server, the language capability is served by a vLLM OpenAI server
(batched, concurrent) with a thin ``OpenAICompatibleBackend`` proxy in front (the proxy
is a normal roboAE msgpack tool service; only the UPSTREAM changes). This removes the
~8-worker/GPU throughput ceiling for the 32B language model.

The vLLM server is NOT a roboAE msgpack service (no identity handshake) -- it is an
OpenAI HTTP server. So it is launched and health-checked here, on a side channel, and
registered with the same owned-process registry the rest of the runtime uses so a run
interrupt tears it down. The proxy tool server (which DOES carry the pinned identity)
still goes through the normal ServiceSupervisor; the proxy's ``load()``/``smoke()`` will
fail loudly if this upstream is not answering, so a broken vLLM launch can never be
silently accepted.

Substrate constant [FIND-s17-3]: ``VLLM_USE_FLASHINFER_SAMPLER=0`` (+
``VLLM_DISABLE_FLASHINFER_PREFILL=1``) is REQUIRED on this CUDA-13/H200/vllm-0.25.1 box or
the engine crashes at warmup in the flashinfer sampling kernel.
"""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


# vLLM launch knobs. gpu_memory_utilization is deliberately below the s17 standalone value (0.85)
# because on a route the vLLM language model (Qwen3-30B-A3B, ~61 GB weights) SHARES its GPU with a
# policy replica (and, on an EGL route, sim
# render is pinned OFF this GPU). s19-H MEASURED that the scaffold's tool requests are tiny (the 32B
# language tool used <=598 tokens; the vision VLM <=1126) and concurrency is <=~16, so the KV cache
# actually needed is only a few GB -- far below what 0.6 (25 GB of KV after the ~61 GB weights)
# reserved. 0.5 of 143 GB (~72 GB) fits the ~61 GB bf16 weights plus a comfortable KV cache for that
# workload and frees ~14 GB for the policy replica / headroom. (Raise it back toward 0.6-0.7 only if a
# CC-revised scaffold pushes concurrency or context far higher.)
VLLM_GPU_MEMORY_UTILIZATION = 0.5
# s19-H: measured max total tokens across routes/resolutions was ~1126 (molmo2-vision at 256x256);
# 2048 keeps ~1.8x headroom for CC-revised scaffolds / higher-res images while ~halving the KV cache
# vs the old 4096. Requests above this hard-fail, so keep the headroom.
VLLM_MAX_MODEL_LEN = 2048
# How long to wait for a vLLM server to finish loading and answer /v1/models. This is a pure BOOT
# WAIT -- it is NOT part of any identity/config_sha256 (that is VLLM_UPSTREAM_TIMEOUT_S below), so it
# only decides how patient we are with a server that is still legitimately loading.
# s20 (2026-07-23): raised 1800 -> 5400. On a FRESH compute node the HF cache reads come off NFS cold,
# and two runs were killed outright by the old 30-minute deadline: the shared vision model was observed
# loading at ~462 s per shard (8 shards ~= 60 min) with two lanes booting at once, and even a single
# language MoE took ~16 min for its 16 shards. 90 minutes covers a cold first boot; a genuinely hung
# server still fails, just later. (Staggering parallel launches remains the real remedy.)
VLLM_READY_TIMEOUT_S = 5400.0
VLLM_POLL_INTERVAL_S = 5.0
# The proxy tool server is launched with --upstream-timeout this value; it is part of the
# openai-compatible identity's config_sha256, so operator_catalog and runtime.py MUST agree.
VLLM_UPSTREAM_TIMEOUT_S = 300.0
# The vLLM upstream port is derived from the proxy tool port to avoid a separate allocator.
VLLM_PORT_STRIDE = 1000


def vllm_served_model_name(model_id: str) -> str:
    """The vLLM --served-model-name for a HF model_id = its last path component (e.g.
    'Qwen/Qwen3-30B-A3B-Instruct-2507' -> 'Qwen3-30B-A3B-Instruct-2507', 'allenai/Molmo2-8B' -> 'Molmo2-8B').
    operator_catalog (identity) and runtime.py (proxy --model + vLLM --served-model-name) both
    derive it this way so the openai-compatible config_sha256 handshake matches."""
    return str(model_id).rsplit("/", 1)[-1]


def openai_runtime_config(service_key: str, served_model_name: str, upstream_timeout_s: float) -> dict:
    """The EXACT dict that tool_services.server._runtime_config produces for a
    `--service <service_key> --runtime openai-compatible --device cpu --model <served>
    --upstream-timeout <T>` launch (with --factory unset). config_sha256 = sha256 of this
    (sorted, compact). Used for BOTH the language (service_key='openai_language') and the vision
    (service_key='openai_vision') tools; operator_catalog pins the tool identity to this hash and
    runtime.py launches the proxy with matching args, so the identity handshake holds."""
    return {
        "service": service_key,
        "runtime": "openai-compatible",
        "device_type": "cpu",
        "upstream_model": served_model_name,
        "factory": None,
        "upstream_timeout_s": float(upstream_timeout_s),
    }


# vLLM gpu-memory-utilization for a vision VLM (Molmo2-8B / Qwen3-VL-8B): it SHARES the vision GPU
# with the pointing/detection/segmentation tools + a policy replica + sim render, so it is capped
# well below the language server's fraction. s19-H measured vision requests at <=1126 tokens and
# concurrency <=~16, needing only ~1-2 GB of KV, so 0.35 (34 GB of KV after the ~16 GB weights) was
# far over-provisioned on the already-crowded vision GPU. 0.25 of 143 GB (~36 GB) holds the ~16 GB 8B
# VLM + an ample KV cache for that workload and frees ~14 GB back to the co-located tools/policy.
VLLM_VISION_GPU_MEMORY_UTILIZATION = 0.25


@dataclass(frozen=True)
class VllmLaunchSpec:
    python: Path            # the vllm conda-env python
    model_path: Path        # the pinned local HF snapshot dir
    served_model_name: str  # what the OpenAI API advertises (== proxy --model)
    gpu_id: int             # physical GPU to pin
    device_uuid: str        # verified physical UUID (defense-in-depth)
    port: int               # the OpenAI server port (proxy points here)
    log_path: Path
    gpu_memory_utilization: float = VLLM_GPU_MEMORY_UTILIZATION
    max_model_len: int = VLLM_MAX_MODEL_LEN


class VllmServer:
    """Owns one vLLM OpenAI server subprocess and its readiness/teardown."""

    def __init__(self, spec: VllmLaunchSpec) -> None:
        self.spec = spec
        self._process: subprocess.Popen[bytes] | None = None
        self._log = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.spec.port}/v1"

    def _env(self) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": str(self.spec.gpu_id),
            # REQUIRED on this substrate (see module docstring).
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_DISABLE_FLASHINFER_PREFILL": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "PATH": f"{self.spec.python.parent}:/usr/bin:/bin",
            "HOME": str(self.spec.log_path.parent),
            "PYTHONNOUSERSITE": "1",
        }

    def _command(self) -> list[str]:
        return [
            str(self.spec.python),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(self.spec.model_path),
            "--served-model-name",
            self.spec.served_model_name,
            "--gpu-memory-utilization",
            f"{self.spec.gpu_memory_utilization:.3f}",
            "--max-model-len",
            str(self.spec.max_model_len),
            "--enforce-eager",
            "--disable-log-stats",
            # The VISION VLMs (Molmo2-8B, Qwen3-VL-8B) ship custom modeling code in their checkpoint
            # and vLLM refuses to load them without this flag; the LANGUAGE model
            # (Qwen3-30B-A3B-Instruct-2507, a standard Qwen3-MoE arch) has no custom code so this is a
            # no-op for it. model_path is a PINNED local HF snapshot
            # (checkpoint_revision), so the executed remote code is the exact validated bytes -- the
            # same code the transformers Molmo2 backend already runs (remote_code_origin=snapshot).
            # This is an UPSTREAM-server flag only; it does not enter the proxy's config_sha256, so the
            # tool-identity handshake is unaffected.
            "--trust-remote-code",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.spec.port),
        ]

    def _models_ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=5.0) as response:
                body = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            return False
        return self.spec.served_model_name in body

    def start(self) -> None:
        from robot_auto_evolve.process_lifecycle import register_owned_process

        if self._process is not None:
            raise RuntimeError("vLLM server already started")
        if not self.spec.python.is_file():
            raise FileNotFoundError(f"vLLM env python is missing: {self.spec.python}")
        if not self.spec.model_path.is_dir():
            raise FileNotFoundError(f"vLLM model snapshot is missing: {self.spec.model_path}")
        self.spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.spec.log_path.open("ab", buffering=0)
        self._process = subprocess.Popen(
            self._command(),
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=self._log,
            env=self._env(),
            start_new_session=True,
        )
        register_owned_process(self._process, f"vllm:{self.spec.served_model_name}:gpu{self.spec.gpu_id}")
        deadline = time.monotonic() + VLLM_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                tail = ""
                try:
                    tail = self.spec.log_path.read_text(errors="replace")[-2000:]
                except OSError:
                    pass
                raise RuntimeError(
                    f"vLLM server exited with code {self._process.returncode} before ready; log tail:\n{tail}"
                )
            if self._models_ready():
                return
            time.sleep(VLLM_POLL_INTERVAL_S)
        self.stop()
        raise TimeoutError(f"vLLM server not ready within {VLLM_READY_TIMEOUT_S:.0f}s")

    def stop(self) -> None:
        import os
        import signal

        from robot_auto_evolve.process_lifecycle import unregister_owned_process

        process = self._process
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=20.0)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=30.0)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        pass
            unregister_owned_process(process)
            self._process = None
        if self._log is not None:
            self._log.close()
            self._log = None
