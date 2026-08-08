from __future__ import annotations

import json
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from robot_auto_evolve.config import Profile, ServiceEndpointProfile
from robot_auto_evolve.benchmarks.simpler_worker import validate_simpler_source
from robot_auto_evolve.benchmarks.openvla_simpler_worker import is_openvla_simpler_adapter
from robot_auto_evolve.policies.config import PolicyServiceConfig
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.services import (
    MsgpackServiceClient,
    ServiceProcessSpec,
    ServiceSupervisor,
)
from robot_auto_evolve.tool_selection import (
    HEAVY_CAPABILITY,
    POLICY_CAPABILITY,
    SERVED_CAPABILITIES,
    TOOL_CAPABILITIES,
    capabilities_for_source,
)
from robot_auto_evolve.tool_services.vllm_launcher import (
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_PORT_STRIDE,
    VLLM_UPSTREAM_TIMEOUT_S,
    VLLM_VISION_GPU_MEMORY_UTILIZATION,
    VllmLaunchSpec,
    VllmServer,
    vllm_served_model_name,
)
from robot_auto_evolve.runtime_paths import (
    RuntimePaths,
    assert_run_runtime_path,
    clean_import_environment,
    project_root_from_package,
    verify_python_import_origin,
)


_TOOL_LAUNCH = {
    "qwen-language": ("qwen_language", "language"),
    "qwen-vision": ("qwen_vision", "vision"),
    "molmo2-vision": ("molmo2_vision", "molmo2"),
    "molmo2-pointing": ("molmo2_pointing", "molmo2"),
    "grounding-dino": ("grounding_dino", "grounding_dino"),
    "sam3": ("sam3", "sam3"),
    "graspgen": ("graspgen", "graspgen"),
    "openai-compatible-language": ("openai_language", "core"),
    "openai-compatible-vision": ("openai_vision", "core"),
}
_VLLM_PROXY_SERVICES = {"openai-compatible-language", "openai-compatible-vision"}

_TOOL_STARTUP_TIMEOUTS = {
    "grounding-dino": 3600.0,
    "qwen-language": 3600.0,
    "qwen-vision": 3600.0,
    "molmo2-vision": 3600.0,
    "molmo2-pointing": 3600.0,
    "sam3": 3600.0,
    "graspgen": 3600.0,
    "openai-compatible-language": 3600.0,
    "openai-compatible-vision": 3600.0,
}

_TOOL_SOURCE = {
    "molmo2-vision": "molmo2",
    "molmo2-pointing": "molmo2",
    "sam3": "sam3",
    "graspgen": "graspgen",
}

_POLICY_LAUNCH = {
    "molmoact2": ("molmoact2", "lerobot_molmoact2_inference"),
    "molmoact2_droid": ("molmoact2", "molmoact2"),
    "molmobot": ("molmobot", "molmobot"),
    "openvla": ("openvla", "simpler_env_openvla"),
    "openpi_droid_jointpos": ("openpi_robolab", "robolab_openpi"),
    "pi05": ("pi05", "lerobot"),
    "smolvla": ("pi05", "lerobot"),
    "rlinf_pi05": ("rlinf_pi05", "rlinf"),
    "rldx": ("rldx", "rldx_1"),
    "xvla": ("xvla", "x_vla"),
}

_POLICY_STARTUP_TIMEOUTS = {
    "molmoact2": 3600.0,
    "molmoact2_droid": 3600.0,
    "molmobot": 3600.0,
    "pi05": 3600.0,
    "smolvla": 3600.0,
    "rlinf_pi05": 3600.0,
    "rldx": 3600.0,
    "openvla": 3600.0,
    "openpi_droid_jointpos": 3600.0,
}

_SOURCE_DIRECTORIES = {
    "calvin": "calvin",
    "graspgen": "graspgen",
    "lerobot": "lerobot",
    "lerobot_molmoact2_inference": "lerobot-molmoact2-inference",
    "libero": "LIBERO",
    "libero_pro": "libero_pro",
    "molmo2": "molmo2",
    "molmoact2": "molmoact2",
    "molmobot": "molmobot",
    "rldx_1": "rldx_1",
    "rlinf": "rlinf",
    "rlinf_lerobot": "rlinf_lerobot",
    "rlinf_openpi": "rlinf_openpi",
    "robocasa365": "robocasa365",
    "robocerebra": "robocerebra",
    "robolab_openpi": "robolab_openpi",
    "robosuite": "robosuite",
    "robotwin_2": "robotwin_2",
    "rrt_algorithms": "rrt_algorithms",
    "sam3": "sam3",
    "simpler_env": "simpler_env",
    "simpler_env_openvla": "simpler_env_openvla",
    "transformers_pi": "transformers-pi",
    "vlabench": "vlabench",
    "x_vla": "X-VLA",
}

_SAPIEN_SUITES = ("simpler_", "robotwin2_")

_GRASPGEN_GRIPPER = "franka"

_PI05_COMPILE_CACHE_SCHEMA = "torch2.7.1-cu126-sm90-v1"
_PI05_COMPILE_THREADS = "20"
_RLINF_PI05_COMPILE_CACHE_SCHEMA = "torch2.7.1-cu126-sm90-v1"
_RLINF_PI05_COMPILE_THREADS = "20"
_NVIDIA_SMI_TIMEOUT_SECONDS = 5.0


def renders_with_egl(suite: str) -> bool:
    return not str(suite).startswith(_SAPIEN_SUITES)


def _endpoint(value: str) -> tuple[str, int]:
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.port is None:
        raise StrictSchemaError("service launcher requires an explicit local HTTP endpoint")
    return parsed.hostname, parsed.port


def _gpu_uuid(gpu_id: int) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "--id", str(gpu_id)],
            check=False,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"GPU {gpu_id} UUID query exceeded five seconds") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"cannot resolve UUID for GPU {gpu_id}: {result.stderr.strip()}")
    return result.stdout.strip()


def _preflight_gpus(gpu_ids: set[int]) -> dict[int, str]:
    uuids = {gpu_id: _gpu_uuid(gpu_id) for gpu_id in sorted(gpu_ids)}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("GPU process query exceeded five seconds") from exc
    if result.returncode != 0:
        raise RuntimeError(f"cannot inspect GPU processes: {result.stderr.strip()}")
    target_uuids = set(uuids.values())
    busy = [line.strip() for line in result.stdout.splitlines() if line.split(",", 1)[0].strip() in target_uuids]
    if busy:
        raise RuntimeError(f"target GPU already has a compute process: {busy[0]}")
    return uuids


def _preflight_ports(services: list[ServiceEndpointProfile]) -> None:
    endpoints = [_endpoint(service.endpoint) for service in services]
    if len(set(endpoints)) != len(endpoints):
        raise RuntimeError("profile contains duplicate service endpoints")
    for host, port in endpoints:
        _preflight_port(host, port)


def _preflight_port(host: str, port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError as exc:
        raise RuntimeError(f"service endpoint is occupied: http://{host}:{port}") from exc
    finally:
        sock.close()


def _wait_for_free_port(host: str, port: int, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            _preflight_port(host, port)
            return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)


def pinned_source(project_root: Path, name: str) -> Path:
    root = Path(project_root).resolve()
    paths = RuntimePaths.load(root)
    directory = _SOURCE_DIRECTORIES.get(name, name)
    source = paths.source(directory)
    if not source.is_dir():
        raise FileNotFoundError(f"pinned source checkout is missing: {source}")
    return source


def service_environment(
    gpu_id: int,
    runtime_root: Path,
    *,
    project_root: Path | None = None,
    paths: RuntimePaths | None = None,
) -> dict[str, str]:
    root = Path(project_root or project_root_from_package()).resolve()
    catalog = paths or RuntimePaths.load(root)
    runtime = assert_run_runtime_path(root, runtime_root)
    temporary = runtime / "tmp"
    home = runtime / "home"
    hf_modules = runtime / "hf_modules"
    temporary.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    hf_modules.mkdir(parents=True, exist_ok=True)
    shared_cache = catalog.shared_cache_root
    environment = {
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "HOME": str(home),
        "HF_HOME": str(shared_cache / "huggingface"),
        "HF_HUB_CACHE": str(catalog.artifact("huggingface_hub")),
        "HF_MODULES_CACHE": str(hf_modules),
        "TORCH_HOME": str(shared_cache / "torch"),
        "XDG_CACHE_HOME": str(shared_cache),
        "PIP_CACHE_DIR": str(shared_cache / "pip"),
        "TMPDIR": str(temporary),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_DISABLED": "true",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_TOKEN": "",
        "HUGGING_FACE_HUB_TOKEN": "",
        "AWS_ACCESS_KEY_ID": "",
        "AWS_SECRET_ACCESS_KEY": "",
        "AWS_SESSION_TOKEN": "",
        "GOOGLE_APPLICATION_CREDENTIALS": "",
        "AZURE_CLIENT_SECRET": "",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "NO_PROXY": "127.0.0.1,localhost",
    }
    environment.update(clean_import_environment(root, catalog))
    return environment


def policy_compile_environment(
    project_root: Path,
    config: PolicyServiceConfig,
    replica_id: str,
    copies_on_gpu: int = 1,
) -> dict[str, str]:
    root = Path(project_root).resolve()
    paths = RuntimePaths.load(root)
    if config.route.backend in {
        "molmoact2",
        "molmoact2_droid",
        "molmobot",
        "openpi_droid_jointpos",
        "openvla",
        "smolvla",
    }:
        return {"MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    compile_settings = {
        "pi05": ("pi05", _PI05_COMPILE_CACHE_SCHEMA, _PI05_COMPILE_THREADS),
        "rlinf_pi05": (
            "rlinf_pi05",
            _RLINF_PI05_COMPILE_CACHE_SCHEMA,
            _RLINF_PI05_COMPILE_THREADS,
        ),
    }.get(config.route.backend)
    if compile_settings is None:
        return {}
    cache_name, cache_schema, compile_threads = compile_settings
    compile_threads = str(max(1, int(compile_threads) // max(1, int(copies_on_gpu))))
    cache = (
        paths.artifact("compile_cache")
        / paths.compile_cache_namespace
        / cache_name
        / cache_schema
        / config.route.name
        / replica_id
    )
    for name in ("cuda", "inductor", "triton"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    environment = {
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "MAX_JOBS": compile_threads,
        "MKL_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        "TORCHINDUCTOR_COMPILE_THREADS": compile_threads,
        "TRITON_CACHE_DIR": str(cache / "triton"),
    }
    if config.route.backend == "rlinf_pi05":
        environment["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
    return environment


def resolve_profile_launch_paths(
    profile: Profile,
    project_root: Path,
    environment_root: Path,
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    catalog = RuntimePaths.load(root)
    environments = Path(environment_root).resolve()
    if environments != catalog.environment_root:
        raise RuntimeError(
            f"environment root differs from runtime_paths.json: {environments} != {catalog.environment_root}"
        )
    if profile.environment.suite == "robocerebra_public60":
        simulator_environment = "libero"
        simulator_source_name = "robocerebra"
        simulator_source_key = "robocerebra_source"
    elif profile.environment.suite.startswith("libero_pro_"):
        simulator_environment = "libero_pro"
        simulator_source_name = "libero_pro"
        simulator_source_key = "libero_pro_source"
    elif profile.environment.suite.startswith("libero"):
        simulator_environment = "libero"
        simulator_source_name = "libero"
        simulator_source_key = "libero_source"
    elif profile.environment.suite.startswith("calvin"):
        simulator_environment = "calvin"
        simulator_source_name = "calvin"
        simulator_source_key = "calvin_source"
    elif profile.environment.suite.startswith("simpler_"):
        if is_openvla_simpler_adapter(profile.environment.adapter):
            simulator_environment = "simpler_openvla"
            simulator_source_name = "simpler_env_openvla"
        else:
            simulator_environment = "simpler_xvla"
            simulator_source_name = None
        simulator_source_key = "simpler_source"
    elif profile.environment.suite == "robotwin2_demo_clean":
        simulator_environment = "robotwin2"
        simulator_source_name = "robotwin_2"
        simulator_source_key = "robotwin2_source"
    elif profile.environment.suite == "vlabench_xvla_tracks_1_4":
        simulator_environment = "vlabench"
        simulator_source_name = "vlabench"
        simulator_source_key = "vlabench_source"
    elif profile.environment.suite == "robocasa365_target":
        simulator_environment = "robocasa365"
        simulator_source_name = "robocasa365"
        simulator_source_key = "robocasa365_source"
    else:
        raise RuntimeError(f"no verified simulator launcher for suite {profile.environment.suite!r}")
    simulator_source = (
        validate_simpler_source(catalog.artifact("simpler_xvla_source"))
        if simulator_source_name is None
        else pinned_source(root, simulator_source_name)
    )
    paths = {
        "agent_python": environments / "agent" / "bin" / "python",
        "simulator_python": environments / simulator_environment / "bin" / "python",
        simulator_source_key: simulator_source,
    }
    if profile.environment.suite == "robotwin2_demo_clean":
        paths["robotwin2_asset_manifest"] = root / "manifests" / "robotwin2_assets.json"
    elif profile.environment.suite == "vlabench_xvla_tracks_1_4":
        paths["vlabench_asset_manifest"] = root / "manifests" / "vlabench_assets.json"
    elif profile.environment.suite == "robocasa365_target":
        paths["robosuite_source"] = pinned_source(root, "robosuite")
        paths["robocasa365_asset_lock"] = catalog.artifact("robocasa365_asset_lock")
    elif profile.environment.suite == "robocerebra_public60":
        paths["robocerebra_asset_manifest"] = root / "manifests" / "robocerebra_assets.json"
        paths["robocerebra_case_catalog"] = root / "manifests" / "robocerebra_cases.json"
        paths["robocerebra_asset_lock"] = catalog.artifact("robocerebra_asset_lock")
    for service in profile.policy.replicas:
        config = PolicyServiceConfig.load(root / "configs" / "policy_services" / f"{service.identity.service_name}.json")
        try:
            environment_name, source_name = _POLICY_LAUNCH[config.route.backend]
        except KeyError as exc:
            raise RuntimeError(f"no verified launcher for policy backend {config.route.backend!r}") from exc
        paths[f"policy_python:{service.identity.replica_id}"] = environments / environment_name / "bin" / "python"
        paths[f"policy_source:{service.identity.replica_id}"] = pinned_source(root, source_name)
        if config.route.backend in {"pi05", "smolvla"}:
            paths[f"policy_transformers_source:{service.identity.replica_id}"] = pinned_source(root, "transformers_pi")
        elif config.route.backend == "rlinf_pi05":
            paths[f"policy_openpi_source:{service.identity.replica_id}"] = pinned_source(root, "rlinf_openpi")
            paths[f"policy_lerobot_source:{service.identity.replica_id}"] = pinned_source(root, "rlinf_lerobot")
    for tool in profile.tools:
        if not tool.enabled or tool.service is None:
            continue
        try:
            _, environment_name = _TOOL_LAUNCH[tool.service.identity.service_name]
        except KeyError as exc:
            raise RuntimeError(f"no verified launcher for tool service {tool.service.identity.service_name!r}") from exc
        paths[f"tool_python:{tool.capability}"] = environments / environment_name / "bin" / "python"
        if tool.service.identity.service_name in _VLLM_PROXY_SERVICES:
            paths["vllm_python"] = environments / "vllm" / "bin" / "python"
        source_name = _TOOL_SOURCE.get(tool.service.identity.service_name)
        if source_name is not None:
            paths[f"tool_source:{tool.capability}"] = pinned_source(root, source_name)
    missing = sorted(str(path) for path in paths.values() if not path.is_file() and not path.is_dir())
    if missing:
        raise FileNotFoundError(f"launch paths are missing: {missing}")
    for python in sorted(
        {path for key, path in paths.items() if key.endswith("_python") or key.startswith("policy_python:") or key.startswith("tool_python:")},
        key=str,
    ):
        verify_python_import_origin(python, root, catalog)
    return paths


@dataclass(frozen=True)
class ScaffoldRuntimePlan:
    capabilities: tuple[str, ...]
    workers: int
    render_gpu_ids: tuple[int, ...]
    render_reason: str
    tool_clients: Mapping[tuple[str, str], MsgpackServiceClient]
    policy_clients: Mapping[tuple[str, str], MsgpackServiceClient]
    policy_replica_ids: tuple[str, ...]

    @property
    def sessions_per_policy(self) -> int:
        if not self.policy_replica_ids:
            return 0
        return (self.workers + len(self.policy_replica_ids) - 1) // len(self.policy_replica_ids)

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "scaffold_runtime_plan",
            "served_tool_capabilities": [
                item for item in self.capabilities if item != POLICY_CAPABILITY
            ],
            "policy_served": POLICY_CAPABILITY in self.capabilities,
            "policy_replica_ids": list(self.policy_replica_ids),
            "workers": self.workers,
            "sessions_per_policy": self.sessions_per_policy,
            "render_gpu_ids": list(self.render_gpu_ids),
            "render_reason": self.render_reason,
        }


class ProfileServiceRuntime:
    def __init__(
        self,
        profile: Profile,
        *,
        project_root: Path,
        environment_root: Path,
        log_root: Path,
    ) -> None:
        if not isinstance(profile, Profile):
            raise StrictSchemaError("service runtime requires Profile")
        self.profile = profile
        self.project_root = Path(project_root).resolve()
        self.runtime_paths = RuntimePaths.load(self.project_root)
        self.environment_root = Path(environment_root).resolve()
        if self.environment_root != self.runtime_paths.environment_root:
            raise RuntimeError("service environment root differs from runtime_paths.json")
        self.log_root = Path(log_root).resolve()
        self.policy_clients: dict[tuple[str, str], MsgpackServiceClient] = {}
        self.policy_supervisors: list[ServiceSupervisor] = []
        self._tool_supervisors: dict[str, ServiceSupervisor] = {}
        self._tool_clients: dict[str, MsgpackServiceClient] = {}
        self._tool_vllm: dict[str, VllmServer] = {}
        self._available_tools = {
            tool.capability: tool
            for tool in profile.tools
            if tool.enabled and tool.service is not None
        }

    @property
    def available_capabilities(self) -> frozenset[str]:
        return frozenset(self._available_tools)

    @property
    def active_capabilities(self) -> frozenset[str]:
        return frozenset(self._tool_supervisors)

    def _policy_spec(self, service: ServiceEndpointProfile) -> ServiceProcessSpec:
        identity = service.identity
        config_path = self.project_root / "configs" / "policy_services" / f"{identity.service_name}.json"
        config = PolicyServiceConfig.load(config_path)
        try:
            environment_name, source_name = _POLICY_LAUNCH[config.route.backend]
        except KeyError as exc:
            raise RuntimeError(f"no verified launcher for policy backend {config.route.backend!r}") from exc
        python = self.environment_root / environment_name / "bin" / "python"
        verify_python_import_origin(python, self.project_root, self.runtime_paths)
        source = pinned_source(self.project_root, source_name)
        if not python.is_file() or not source.is_dir():
            raise FileNotFoundError("policy environment or pinned source checkout is missing")
        host, port = _endpoint(service.endpoint)
        gpu_id = identity.gpu_ids[0]
        command = (
            str(python),
            "-m",
            "robot_auto_evolve.policies.server",
            "--config",
            str(config_path),
            "--source-root",
            str(source),
            "--gpu-id",
            str(gpu_id),
            "--device",
            "cuda:0",
            "--device-uuid",
            _gpu_uuid(gpu_id),
            "--replica-id",
            identity.replica_id,
            "--host",
            host,
            "--port",
            str(port),
        )
        environment = service_environment(
            gpu_id,
            self.log_root / "runtime" / identity.service_name / identity.replica_id,
            project_root=self.project_root,
            paths=self.runtime_paths,
        )
        copies_on_gpu = sum(
            1 for item in self.profile.policy.replicas if item.identity.gpu_ids[0] == gpu_id
        )
        environment.update(
            policy_compile_environment(
                self.project_root,
                config,
                f"{identity.replica_id}-physical-gpu-{gpu_id}",
                copies_on_gpu=copies_on_gpu,
            )
        )
        environment.update(
            {"CONDA_PREFIX": str(python.parent.parent), "PATH": f"{python.parent}:/usr/bin:/bin"}
        )
        return ServiceProcessSpec(
            command,
            self.project_root,
            environment,
            service.endpoint,
            identity,
            startup_timeout_s=_POLICY_STARTUP_TIMEOUTS.get(config.route.backend, 900.0),
        )

    def _hf_snapshot(self, model_id: str, revision: str) -> Path:
        cache = self.runtime_paths.artifact("huggingface_hub")
        snapshot = cache / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
        if not snapshot.is_dir():
            raise FileNotFoundError(f"pinned HF snapshot is missing for the vLLM upstream: {snapshot}")
        return snapshot

    def _vllm_spec(self, service: ServiceEndpointProfile) -> VllmLaunchSpec | None:
        identity = service.identity
        if identity.service_name not in _VLLM_PROXY_SERVICES:
            return None
        _, proxy_port = _endpoint(service.endpoint)
        gpu_id = identity.gpu_ids[0]
        is_vision = identity.service_name == "openai-compatible-vision"
        return VllmLaunchSpec(
            python=self.environment_root / "vllm" / "bin" / "python",
            model_path=self._hf_snapshot(identity.model_id, identity.checkpoint_revision),
            served_model_name=vllm_served_model_name(identity.model_id),
            gpu_id=gpu_id,
            device_uuid=_gpu_uuid(gpu_id),
            port=proxy_port + VLLM_PORT_STRIDE,
            log_path=self.log_root / "vllm" / f"{identity.service_name}-gpu{gpu_id}.log",
            gpu_memory_utilization=(
                VLLM_VISION_GPU_MEMORY_UTILIZATION if is_vision else VLLM_GPU_MEMORY_UTILIZATION
            ),
        )

    def _openai_proxy_spec(self, service: ServiceEndpointProfile) -> ServiceProcessSpec:
        identity = service.identity
        service_key = "openai_vision" if identity.service_name == "openai-compatible-vision" else "openai_language"
        python = self.environment_root / "core" / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(f"core env python is missing for the {service_key} proxy: {python}")
        verify_python_import_origin(python, self.project_root, self.runtime_paths)
        host, proxy_port = _endpoint(service.endpoint)
        gpu_id = identity.gpu_ids[0]
        vllm_port = proxy_port + VLLM_PORT_STRIDE
        identity_dir = self.log_root / "identities"
        identity_dir.mkdir(parents=True, exist_ok=True)
        (identity_dir / f"{identity.service_name}-{identity.replica_id}.json").write_text(
            json.dumps(identity.to_mapping(), sort_keys=True, indent=2) + "\n"
        )
        command = (
            str(python),
            "-m",
            "robot_auto_evolve.tool_services.server",
            "--service",
            service_key,
            "--runtime",
            "openai-compatible",
            "--base-url",
            f"http://127.0.0.1:{vllm_port}/v1",
            "--model",
            vllm_served_model_name(identity.model_id),
            "--model-id",
            identity.model_id,
            "--checkpoint-revision",
            identity.checkpoint_revision,
            "--upstream-timeout",
            f"{VLLM_UPSTREAM_TIMEOUT_S:.1f}",
            "--host",
            host,
            "--port",
            str(proxy_port),
            "--gpu-id",
            str(gpu_id),
            "--device",
            "cpu",
            "--device-uuid",
            f"cpu-proxy-{identity.replica_id}",
            "--replica-id",
            identity.replica_id,
        )
        environment = service_environment(
            gpu_id,
            self.log_root / "runtime" / identity.service_name / identity.replica_id,
            project_root=self.project_root,
            paths=self.runtime_paths,
        )
        environment.update({"CONDA_PREFIX": str(python.parent.parent), "PATH": f"{python.parent}:/usr/bin:/bin"})
        return ServiceProcessSpec(
            command,
            self.project_root,
            environment,
            service.endpoint,
            identity,
            startup_timeout_s=_TOOL_STARTUP_TIMEOUTS[identity.service_name],
        )

    def _tool_spec(self, service: ServiceEndpointProfile) -> ServiceProcessSpec:
        identity = service.identity
        if identity.service_name in _VLLM_PROXY_SERVICES:
            return self._openai_proxy_spec(service)
        try:
            service_arg, environment_name = _TOOL_LAUNCH[identity.service_name]
        except KeyError as exc:
            raise RuntimeError(f"no verified launcher for tool service {identity.service_name!r}") from exc
        if identity.service_name == "graspgen":
            embodiment = self.profile.environment.embodiment
            if _GRASPGEN_GRIPPER not in embodiment.lower():
                raise RuntimeError(
                    f"GraspGen serves only the {_GRASPGEN_GRIPPER} gripper, but this route's robot is "
                    f"{embodiment!r}; its grasp poses would describe a gripper this robot does not have"
                )
            if not any(camera.has_depth for camera in self.profile.environment.cameras):
                raise RuntimeError("GraspGen needs metric depth, and no camera on this route has it")
        python = self.environment_root / environment_name / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(f"tool environment is missing: {python.parent.parent}")
        verify_python_import_origin(python, self.project_root, self.runtime_paths)
        host, port = _endpoint(service.endpoint)
        gpu_id = identity.gpu_ids[0]
        identity_dir = self.log_root / "identities"
        identity_dir.mkdir(parents=True, exist_ok=True)
        identity_path = identity_dir / f"{identity.service_name}-{identity.replica_id}.json"
        identity_path.write_text(json.dumps(identity.to_mapping(), sort_keys=True, indent=2) + "\n")
        command = (
            str(python),
            "-m",
            "robot_auto_evolve.tool_services.server",
            "--service",
            service_arg,
            "--runtime",
            "official" if identity.service_name in {"sam3", "graspgen"} else "transformers",
            "--host",
            host,
            "--port",
            str(port),
            "--gpu-id",
            str(gpu_id),
            "--device",
            "cuda:0",
            "--device-uuid",
            _gpu_uuid(gpu_id),
            "--replica-id",
            identity.replica_id,
            "--identity-json",
            str(identity_path),
        )
        environment = service_environment(
            gpu_id,
            self.log_root / "runtime" / identity.service_name / identity.replica_id,
            project_root=self.project_root,
            paths=self.runtime_paths,
        )
        environment.update(
            {"CONDA_PREFIX": str(python.parent.parent), "PATH": f"{python.parent}:/usr/bin:/bin"}
        )
        return ServiceProcessSpec(
            command,
            self.project_root,
            environment,
            service.endpoint,
            identity,
            startup_timeout_s=_TOOL_STARTUP_TIMEOUTS[identity.service_name],
        )

    def start(self) -> Mapping[tuple[str, str], MsgpackServiceClient]:
        services = list(self.profile.policy.replicas)
        try:
            self.log_root.mkdir(parents=True, exist_ok=True)
            gpu_uuids = _preflight_gpus({gpu_id for gpu_id in self.profile.resources.gpu_ids})
            _preflight_ports(
                services + [tool.service for tool in self._available_tools.values()]
            )
            (self.log_root / "preflight.json").write_text(
                json.dumps(
                    {
                        "checked_ns": time.time_ns(),
                        "gpu_uuids": {str(key): value for key, value in gpu_uuids.items()},
                        "policy_endpoints": [service.endpoint for service in services],
                        "declared_tool_capabilities": sorted(self._available_tools),
                        "note": "no model is started here; each scaffold decides",
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return {}
        except BaseException:
            self.stop()
            raise

    def _stop_tool(self, capability: str) -> None:
        supervisor = self._tool_supervisors.pop(capability, None)
        self._tool_clients.pop(capability, None)
        if supervisor is not None:
            supervisor.stop()
        server = self._tool_vllm.pop(capability, None)
        if server is not None:
            server.stop()

    def _launch_tool(self, capability: str) -> None:
        service = self._available_tools[capability].service
        host, port = _endpoint(service.endpoint)
        _wait_for_free_port(host, port)
        spec = self._vllm_spec(service)
        if spec is not None:
            _wait_for_free_port("127.0.0.1", spec.port)
            server = VllmServer(spec)
            self._tool_vllm[capability] = server
            server.start()
        supervisor = ServiceSupervisor(
            self._tool_spec(service),
            self.log_root / service.identity.service_name,
            reuse_exact=False,
        )
        self._tool_supervisors[capability] = supervisor
        supervisor.launch()

    def _stop_policies(self) -> None:
        for supervisor in reversed(self.policy_supervisors):
            supervisor.stop()
        self.policy_supervisors.clear()
        self.policy_clients.clear()

    def _launch_policies(self) -> None:
        keyed = []
        for service in self.profile.policy.replicas:
            host, port = _endpoint(service.endpoint)
            _wait_for_free_port(host, port)
            supervisor = ServiceSupervisor(
                self._policy_spec(service),
                self.log_root / service.identity.service_name,
                reuse_exact=False,
            )
            self.policy_supervisors.append(supervisor)
            keyed.append(((service.identity.service_name, service.identity.replica_id), supervisor))
        for _, supervisor in keyed:
            supervisor.launch()
        for key, supervisor in keyed:
            self.policy_clients[key] = supervisor.wait_ready()

    @property
    def policy_active(self) -> bool:
        return bool(self.policy_supervisors)

    def ensure_policies(self, wanted: bool) -> Mapping[tuple[str, str], MsgpackServiceClient]:
        wanted = bool(wanted) and bool(self.profile.policy.replicas)
        if wanted and not self.policy_active:
            self._launch_policies()
        elif not wanted and self.policy_active:
            self._stop_policies()
        (self.log_root / "policy_ready.json").write_text(
            json.dumps(
                {
                    "ready_ns": time.time_ns(),
                    "served": wanted,
                    "declared_replica_ids": [
                        service.identity.replica_id for service in self.profile.policy.replicas
                    ],
                    "services": [
                        {
                            "identity": supervisor.spec.identity.to_mapping(),
                            "pid": None if supervisor.process is None else supervisor.process.pid,
                        }
                        for supervisor in self.policy_supervisors
                    ],
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return dict(self.policy_clients)

    def ensure_tools(self, capabilities) -> Mapping[tuple[str, str], MsgpackServiceClient]:
        wanted = frozenset(capabilities) & self.available_capabilities - {POLICY_CAPABILITY}
        for capability in sorted(self.active_capabilities - wanted):
            self._stop_tool(capability)
        starting = sorted(wanted - self.active_capabilities)
        for capability in starting:
            self._launch_tool(capability)
        for capability in starting:
            self._tool_clients[capability] = self._tool_supervisors[capability].wait_ready()
        (self.log_root / "tools_ready.json").write_text(
            json.dumps(
                {
                    "ready_ns": time.time_ns(),
                    "served": sorted(wanted),
                    "declared": sorted(self.available_capabilities),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        result = {}
        for capability in sorted(wanted):
            identity = self._available_tools[capability].service.identity
            result[(identity.service_name, identity.replica_id)] = self._tool_clients[capability]
        return result

    def stop(self) -> None:
        for capability in sorted(self._tool_supervisors):
            self._stop_tool(capability)
        self._tool_supervisors.clear()
        self._tool_clients.clear()
        for server in list(self._tool_vllm.values()):
            server.stop()
        self._tool_vllm.clear()
        self._stop_policies()

    def __enter__(self) -> Mapping[tuple[str, str], MsgpackServiceClient]:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


class ScaffoldRuntimeCoordinator:
    def __init__(
        self,
        runtime: ProfileServiceRuntime,
        *,
        gpu_ids,
        render_gpu_ids_override,
        workers_per_gpu: int,
        workers_per_gpu_with_language: int,
        policies_per_gpu: int = 1,
        egl: bool,
    ) -> None:
        self.runtime = runtime
        self.gpu_ids = tuple(int(item) for item in gpu_ids)
        self.render_gpu_ids_override = (
            None if render_gpu_ids_override is None else tuple(int(item) for item in render_gpu_ids_override)
        )
        if self.render_gpu_ids_override is not None and len(self.render_gpu_ids_override) != len(self.gpu_ids):
            raise StrictSchemaError("render GPU override needs one entry per pool GPU")
        if workers_per_gpu < 1 or workers_per_gpu_with_language < 1:
            raise StrictSchemaError("workers per GPU must be positive")
        self.workers_per_gpu = int(workers_per_gpu)
        self.workers_per_gpu_with_language = int(workers_per_gpu_with_language)
        self.policies_per_gpu = int(policies_per_gpu)
        self.egl = bool(egl)

    def capabilities_for(self, scaffold_source: str) -> frozenset[str]:
        served = self.runtime.available_capabilities | {POLICY_CAPABILITY}
        return capabilities_for_source(scaffold_source) & served

    def render_gpu_ids_for(self, capabilities) -> tuple[tuple[int, ...], str]:
        if self.render_gpu_ids_override is not None:
            return self.render_gpu_ids_override, "explicit --render-gpu-ids"
        if self.egl:
            pinned = tuple(self.gpu_ids[-1] for _ in self.gpu_ids)
            return pinned, f"MuJoCo-EGL route: every episode renders on the last pool GPU ({self.gpu_ids[-1]})"
        return self.gpu_ids, "SAPIEN/Vulkan route: episodes render round-robin, one GPU each"

    def workers_for(self, capabilities) -> int:
        per_gpu = (
            self.workers_per_gpu_with_language
            if HEAVY_CAPABILITY in capabilities
            else self.workers_per_gpu
        )
        return len(self.gpu_ids) * per_gpu

    def plan_for(self, scaffold_source: str) -> ScaffoldRuntimePlan:
        capabilities = self.capabilities_for(scaffold_source)
        policy_wanted = POLICY_CAPABILITY in capabilities
        if policy_wanted:
            clients = self.runtime.ensure_tools(capabilities)
            policy_clients = self.runtime.ensure_policies(True)
        else:
            policy_clients = self.runtime.ensure_policies(False)
            clients = self.runtime.ensure_tools(capabilities)
        render_gpu_ids, reason = self.render_gpu_ids_for(capabilities)
        return ScaffoldRuntimePlan(
            capabilities=tuple(sorted(capabilities)),
            workers=self.workers_for(capabilities),
            render_gpu_ids=render_gpu_ids,
            render_reason=reason,
            tool_clients=clients,
            policy_clients=policy_clients,
            policy_replica_ids=tuple(replica_id for _, replica_id in sorted(policy_clients)),
        )


__all__ = [
    "ProfileServiceRuntime",
    "ScaffoldRuntimeCoordinator",
    "ScaffoldRuntimePlan",
    "TOOL_CAPABILITIES",
    "pinned_source",
    "policy_compile_environment",
    "renders_with_egl",
    "resolve_profile_launch_paths",
    "service_environment",
]
