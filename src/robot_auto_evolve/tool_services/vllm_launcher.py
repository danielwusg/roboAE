
from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


VLLM_GPU_MEMORY_UTILIZATION = 0.5
VLLM_MAX_MODEL_LEN = 2048
VLLM_READY_TIMEOUT_S = 5400.0
VLLM_POLL_INTERVAL_S = 5.0
VLLM_UPSTREAM_TIMEOUT_S = 300.0
VLLM_PORT_STRIDE = 1000


def vllm_served_model_name(model_id: str) -> str:
    return str(model_id).rsplit("/", 1)[-1]


def openai_runtime_config(service_key: str, served_model_name: str, upstream_timeout_s: float) -> dict:
    return {
        "service": service_key,
        "runtime": "openai-compatible",
        "device_type": "cpu",
        "upstream_model": served_model_name,
        "factory": None,
        "upstream_timeout_s": float(upstream_timeout_s),
    }


VLLM_VISION_GPU_MEMORY_UTILIZATION = 0.25


@dataclass(frozen=True)
class VllmLaunchSpec:
    python: Path
    model_path: Path
    served_model_name: str
    gpu_id: int
    device_uuid: str
    port: int
    log_path: Path
    gpu_memory_utilization: float = VLLM_GPU_MEMORY_UTILIZATION
    max_model_len: int = VLLM_MAX_MODEL_LEN


class VllmServer:

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
