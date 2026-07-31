from __future__ import annotations

import json
import socket
import subprocess
import time
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
    RuntimeArtifactLock,
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

_PI05_COMPILE_CACHE_SCHEMA = "torch2.7.1-cu126-sm90-v1"
_PI05_COMPILE_THREADS = "20"
_RLINF_PI05_COMPILE_CACHE_SCHEMA = "torch2.7.1-cu126-sm90-v1"
_RLINF_PI05_COMPILE_THREADS = "20"
_NVIDIA_SMI_TIMEOUT_SECONDS = 5.0


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
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"service endpoint is occupied: http://{host}:{port}") from exc
        finally:
            sock.close()


def locked_source(project_root: Path, name: str) -> Path:
    root = Path(project_root).resolve()
    paths = RuntimePaths.load(root)
    entry = RuntimeArtifactLock.load(root).source(name)
    directory = entry.get("directory", name)
    expected_commit = entry["commit"]
    source = paths.source(directory)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected_commit:
        raise RuntimeError(f"source revision mismatch for {name}: {actual}")
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(source),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"source working tree is dirty for {name}: {dirty.splitlines()[0]}")
    submodules = entry.get("submodules", {})
    if not isinstance(submodules, dict):
        raise StrictSchemaError(f"source lock has invalid submodules for {name!r}")
    for relative, expected in sorted(submodules.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise StrictSchemaError(f"source lock has invalid submodule pin for {name!r}")
        submodule = (source / relative).resolve()
        try:
            submodule.relative_to(source)
        except ValueError as exc:
            raise StrictSchemaError(f"source lock submodule escapes checkout for {name!r}") from exc
        if not submodule.is_dir():
            raise FileNotFoundError(f"pinned submodule checkout is missing: {submodule}")
        actual_submodule = subprocess.check_output(
            ["git", "-C", str(submodule), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual_submodule != expected:
            raise RuntimeError(f"submodule revision mismatch for {name}:{relative}: {actual_submodule}")
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
    cache = (
        paths.artifact("compile_cache")
        / paths.compile_cache_namespace
        / cache_name
        / cache_schema
        / config.sha256
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
        validate_simpler_source(catalog.artifact("simpler_xvla_source"), full_tree=True)
        if simulator_source_name is None
        else locked_source(root, simulator_source_name)
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
        paths["robosuite_source"] = locked_source(root, "robosuite")
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
        paths[f"policy_source:{service.identity.replica_id}"] = locked_source(root, source_name)
        if config.route.backend in {"pi05", "smolvla"}:
            paths[f"policy_transformers_source:{service.identity.replica_id}"] = locked_source(root, "transformers_pi")
        elif config.route.backend == "rlinf_pi05":
            paths[f"policy_openpi_source:{service.identity.replica_id}"] = locked_source(root, "rlinf_openpi")
            paths[f"policy_lerobot_source:{service.identity.replica_id}"] = locked_source(root, "rlinf_lerobot")
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
            paths[f"tool_source:{tool.capability}"] = locked_source(root, source_name)
    missing = sorted(str(path) for path in paths.values() if not path.is_file() and not path.is_dir())
    if missing:
        raise FileNotFoundError(f"launch paths are missing: {missing}")
    for python in sorted(
        {path for key, path in paths.items() if key.endswith("_python") or key.startswith("policy_python:") or key.startswith("tool_python:")},
        key=str,
    ):
        verify_python_import_origin(python, root, catalog)
    return paths


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
        self.clients: dict[tuple[str, str], MsgpackServiceClient] = {}
        self.supervisors: list[ServiceSupervisor] = []
        self._vllm_servers: list[VllmServer] = []

    def _policy_spec(self, service: ServiceEndpointProfile) -> ServiceProcessSpec:
        identity = service.identity
        config_path = self.project_root / "configs" / "policy_services" / f"{identity.service_name}.json"
        config = PolicyServiceConfig.load(config_path)
        if config.sha256 != identity.config_sha256:
            raise StrictSchemaError("policy launch config hash differs from profile identity")
        try:
            environment_name, source_name = _POLICY_LAUNCH[config.route.backend]
        except KeyError as exc:
            raise RuntimeError(f"no verified launcher for policy backend {config.route.backend!r}") from exc
        python = self.environment_root / environment_name / "bin" / "python"
        verify_python_import_origin(python, self.project_root, self.runtime_paths)
        source = locked_source(self.project_root, source_name)
        if config.route.backend in {"pi05", "smolvla"}:
            locked_source(self.project_root, "transformers_pi")
        elif config.route.backend == "rlinf_pi05":
            locked_source(self.project_root, "rlinf_openpi")
            locked_source(self.project_root, "rlinf_lerobot")
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
        environment.update(
            policy_compile_environment(
                self.project_root,
                config,
                f"{identity.replica_id}-physical-gpu-{gpu_id}",
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

    def _vllm_specs(self) -> list[VllmLaunchSpec]:
        specs: list[VllmLaunchSpec] = []
        for tool in self.profile.tools:
            if not tool.enabled or tool.service is None:
                continue
            identity = tool.service.identity
            if identity.service_name not in _VLLM_PROXY_SERVICES:
                continue
            _, proxy_port = _endpoint(tool.service.endpoint)
            gpu_id = identity.gpu_ids[0]
            is_vision = identity.service_name == "openai-compatible-vision"
            specs.append(
                VllmLaunchSpec(
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
            )
        return specs

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
        python = self.environment_root / environment_name / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(f"tool environment is missing: {python.parent.parent}")
        verify_python_import_origin(python, self.project_root, self.runtime_paths)
        source_name = _TOOL_SOURCE.get(identity.service_name)
        if source_name is not None:
            locked_source(self.project_root, source_name)
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
        services.extend(
            tool.service
            for tool in self.profile.tools
            if tool.enabled and tool.service is not None
        )
        try:
            self.log_root.mkdir(parents=True, exist_ok=True)
            gpu_uuids = _preflight_gpus(
                {gpu_id for service in services for gpu_id in service.identity.gpu_ids}
            )
            _preflight_ports(services)
            vllm_specs = self._vllm_specs()
            for spec in vllm_specs:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(("127.0.0.1", spec.port))
                except OSError as exc:
                    raise RuntimeError(f"vLLM upstream port is occupied: http://127.0.0.1:{spec.port}") from exc
                finally:
                    sock.close()
            for spec in vllm_specs:
                server = VllmServer(spec)
                self._vllm_servers.append(server)
                server.start()
            (self.log_root / "preflight.json").write_text(
                json.dumps(
                    {
                        "checked_ns": time.time_ns(),
                        "gpu_uuids": {str(key): value for key, value in gpu_uuids.items()},
                        "service_endpoints": [service.endpoint for service in services],
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            keyed_supervisors = []
            for service in services:
                key = (service.identity.service_name, service.identity.replica_id)
                spec = self._policy_spec(service) if service.identity.service_kind == "policy" else self._tool_spec(service)
                supervisor = ServiceSupervisor(
                    spec,
                    self.log_root / service.identity.service_name,
                    reuse_exact=False,
                )
                self.supervisors.append(supervisor)
                keyed_supervisors.append((key, supervisor))
            for _, supervisor in keyed_supervisors:
                supervisor.launch()
            for key, supervisor in keyed_supervisors:
                self.clients[key] = supervisor.wait_ready()
            self.profile.validate_service_identities(
                [client.validate_identity() for client in self.clients.values()]
            )
            (self.log_root / "services_ready.json").write_text(
                json.dumps(
                    {
                        "ready_ns": time.time_ns(),
                        "services": [
                            {
                                "identity": supervisor.spec.identity.to_mapping(),
                                "pid": None if supervisor.process is None else supervisor.process.pid,
                                "reused": supervisor.reused,
                            }
                            for _, supervisor in keyed_supervisors
                        ],
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return dict(self.clients)
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        for supervisor in reversed(self.supervisors):
            supervisor.stop()
        self.supervisors.clear()
        self.clients.clear()
        for server in reversed(self._vllm_servers):
            server.stop()
        self._vllm_servers.clear()

    def __enter__(self) -> Mapping[tuple[str, str], MsgpackServiceClient]:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()
