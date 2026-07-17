from __future__ import annotations

import importlib
import importlib.util
import hashlib
import os
import random
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_auto_evolve.config import Profile
from robot_auto_evolve.protocol import (
    CameraObservation,
    CanonicalActionChunk,
    FairObservation,
    RobotProprioception,
    RobotStateVector,
    StrictSchemaError,
)
from robot_auto_evolve.provenance import EpisodeKey

from .robotwin2_assets import read_and_validate_robotwin2_asset_record
from .robotwin2_protocol import (
    ROBOTWIN2_BENCHMARK_PROTOCOL,
    ROBOTWIN2_BENCHMARK_SMOKE_PROTOCOL,
    ROBOTWIN2_RELATED_PAIRS,
    ROBOTWIN2_RELATED_PROTOCOL,
    ROBOTWIN2_RELATED_SMOKE_PROTOCOL,
    ROBOTWIN2_SCENARIO,
    ROBOTWIN2_SOURCE_COMMIT,
    expected_horizon,
    validate_official_step_limits,
)
from .xvla import ROBOTWIN_ACTION_SPEC, ROBOTWIN_TASKS


ROBOTWIN2_CONFIG = "demo_clean"
ROBOTWIN2_EMBODIMENT = ("aloha-agilex",)
CUROBO_TASK_CONFIG_INPUT_SHA256 = {
    "base_cfg.yml": "82b7c484a7f08ec0a0696a7177972b8075d8cbfdf35343c636a101f08d0da6a7",
    "finetune_trajopt.yml": "0811d6f8afc362c76b783ef1a5e548047d15eb714bf630238c608c05287bfa14",
    "finetune_trajopt_slow.yml": "de54f3b219b202f0311147ca9948ae709cd00811b52c9b82221147cfa4e95aca",
    "gradient_ik.yml": "c83954f640a137ae8db0bc3370eef387cf67cb12fe9e9fd05f71f331ca3ea32b",
    "gradient_ik_autotune.yml": "ad3f34da4f8f7e9b699ffb84e2cb2bdeb5a6b693dfbab715c4bd10400c51db1c",
    "gradient_mpc.yml": "c7773e5e65c2c16e76f829c0e304ee367abfbaedb6dabf12054d4f363ac434a6",
    "gradient_trajopt.yml": "6533d27ce6b8dfecdf007158fea57a35b5c285f43958b85e27fabf776ff78b42",
    "graph.yml": "05be858a85555a4abc9ed445c5451b524b36965305cbfbcd6a8cfc6da9f04186",
    "particle_ik.yml": "e5f7e9b81860a319dff90507698db6ccfa885ec551c29f26c8f88410e23076a2",
    "particle_mpc.yml": "3ef3ae6aea7c51e3fef79ad45c283e37471e5b9a07ff5acb0652a251ca56536b",
    "particle_trajopt.yml": "18ca9253c8e38b5eedb0b82de51fd229ad1cb8d9b3ed3acc414108031db1012c",
}
CUROBO_TASK_CONFIG_OUTPUT_SHA256 = {
    **CUROBO_TASK_CONFIG_INPUT_SHA256,
    "finetune_trajopt.yml": "560dcc86c2fc8377b25c662d35e63532132f295d6510704f93aac1fa1b9787c8",
    "finetune_trajopt_slow.yml": "2943e3e181edf18290a8c005e4e94c21e18e85b979d49e996088da52bcc5fae7",
    "gradient_ik_autotune.yml": "57b1eed48bf1a1f592d857bcfffc44003a1fe497808eb746d143665a414034da",
    "gradient_trajopt.yml": "4de1d28b5d840d9f3d5a73541ccea0af06c23ca8f29b86d077e2474ec69f3bf0",
}
CUROBO_COMPATIBILITY_CONFIGS = frozenset(
    {
        "finetune_trajopt.yml",
        "finetune_trajopt_slow.yml",
        "gradient_ik_autotune.yml",
        "gradient_trajopt.yml",
    }
)
CUROBO_COMPATIBILITY_KEYS = (
    "use_cuda_kernel",
    "use_cuda_line_search_kernel",
    "use_cuda_update_best_kernel",
)
_CUROBO_TASK_CONFIG_OVERLAY: Path | None = None
SAPIEN_VERSION = "3.0.0b1"
SAPIEN_WRAPPER_SHA256 = {
    "sapien.wrapper.engine": "16acbd34e2fb2f2e03c6df77ef1a7211a0765c037b4fa7a6e8c36975ca19a813",
    "sapien.wrapper.renderer": "af0990f96331191e6ff8c260f8126ea8f81086b2376807b175fecc0ec405dbc2",
    "sapien.wrapper.scene": "9b318264e83475534d56701191ae055b366d4245b2e4e535e90b3e1ffaa01e46",
}
_SAPIEN_RENDER_GPU_ID: int | None = None
_SAPIEN_RENDER_PCI: str | None = None
_SAPIEN_ROUTED_RENDERER: Any = None
_SAPIEN_ROUTED_CREATE_SCENE: Any = None


def _write_runtime_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise RuntimeError(f"RoboTwin 2 runtime file differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _disable_curobo_optimizer_kernels(name: str, payload: bytes) -> bytes:
    parsed = yaml.safe_load(payload)
    lbfgs = parsed.get("lbfgs") if isinstance(parsed, dict) else None
    if not isinstance(lbfgs, dict) or any(lbfgs.get(key) is not True for key in CUROBO_COMPATIBILITY_KEYS):
        raise RuntimeError(f"cuRobo task config optimizer flags differ: {name}")
    rendered = payload
    for key in CUROBO_COMPATIBILITY_KEYS:
        candidates = (f"  {key}: True\n".encode(), f"  {key}: true\n".encode())
        matches = sum(rendered.count(candidate) for candidate in candidates)
        if matches != 1:
            raise RuntimeError(f"cuRobo task config flag encoding differs: {name}:{key}")
        for candidate in candidates:
            rendered = rendered.replace(candidate, f"  {key}: false\n".encode())
    result = yaml.safe_load(rendered)
    result_lbfgs = result.get("lbfgs") if isinstance(result, dict) else None
    if not isinstance(result_lbfgs, dict) or any(
        result_lbfgs.get(key) is not False for key in CUROBO_COMPATIBILITY_KEYS
    ):
        raise RuntimeError(f"cuRobo task config compatibility flags differ: {name}")
    return rendered


def _runtime_curobo_task_configs(upstream: Path) -> Path:
    temporary_value = os.environ.get("TMPDIR")
    if not temporary_value:
        raise RuntimeError("RoboTwin 2 worker requires TMPDIR")
    temporary = Path(temporary_value).resolve()
    upstream = upstream.resolve()
    if not temporary.is_dir() or not upstream.is_dir() or upstream.is_symlink():
        raise RuntimeError("cuRobo task config source or TMPDIR is invalid")
    entries = {path.name for path in upstream.iterdir()}
    if entries != set(CUROBO_TASK_CONFIG_INPUT_SHA256):
        raise RuntimeError("cuRobo task config file set differs")
    overlay = temporary / "robotwin2_curobo_task_configs_v078"
    overlay.mkdir(parents=True, exist_ok=True)
    if overlay.is_symlink() or not overlay.is_dir():
        raise RuntimeError("cuRobo task config overlay is invalid")
    for name, input_hash in CUROBO_TASK_CONFIG_INPUT_SHA256.items():
        source = upstream / name
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"cuRobo task config source is invalid: {name}")
        payload = source.read_bytes()
        if _sha256(payload) != input_hash:
            raise RuntimeError(f"cuRobo task config source hash differs: {name}")
        if name in CUROBO_COMPATIBILITY_CONFIGS:
            payload = _disable_curobo_optimizer_kernels(name, payload)
        if _sha256(payload) != CUROBO_TASK_CONFIG_OUTPUT_SHA256[name]:
            raise RuntimeError(f"cuRobo task config output hash differs: {name}")
        _write_runtime_file(overlay / name, payload)
    output_entries = {path.name for path in overlay.iterdir()}
    if output_entries != set(CUROBO_TASK_CONFIG_OUTPUT_SHA256):
        raise RuntimeError("cuRobo task config overlay file set differs")
    _validate_curobo_task_config_overlay(overlay)
    overlay.chmod(0o555)
    return overlay


def _validate_curobo_task_config_overlay(overlay: Path) -> None:
    output_entries = {path.name for path in overlay.iterdir()}
    if output_entries != set(CUROBO_TASK_CONFIG_OUTPUT_SHA256):
        raise RuntimeError("cuRobo task config overlay file set differs")
    for name, output_hash in CUROBO_TASK_CONFIG_OUTPUT_SHA256.items():
        path = overlay / name
        if not path.is_file() or path.is_symlink() or _sha256(path.read_bytes()) != output_hash:
            raise RuntimeError(f"cuRobo task config overlay differs: {name}")


def _install_runtime_curobo_task_configs() -> Path:
    global _CUROBO_TASK_CONFIG_OVERLAY
    util_file = importlib.import_module("curobo.util_file")
    if _CUROBO_TASK_CONFIG_OVERLAY is not None:
        overlay = _CUROBO_TASK_CONFIG_OVERLAY
        if Path(util_file.get_task_configs_path()).resolve() != overlay:
            raise RuntimeError("cuRobo task config overlay changed within one process")
        _validate_curobo_task_config_overlay(overlay)
        return overlay
    consumers = {
        name
        for name in sys.modules
        if name
        in {
            "curobo.graph.graph_base",
            "curobo.wrap.reacher.ik_solver",
            "curobo.wrap.reacher.motion_gen",
            "curobo.wrap.reacher.mpc",
            "curobo.wrap.reacher.trajopt",
        }
    }
    if consumers:
        raise RuntimeError(f"cuRobo task config consumers were imported too early: {sorted(consumers)}")
    upstream = Path(util_file.get_task_configs_path()).resolve()
    overlay = _runtime_curobo_task_configs(upstream)

    def task_config_path() -> str:
        return str(overlay)

    util_file.get_task_configs_path = task_config_path
    if Path(util_file.get_task_configs_path()).resolve() != overlay:
        raise RuntimeError("cuRobo task config path installation failed")
    _CUROBO_TASK_CONFIG_OVERLAY = overlay
    return overlay


def _runtime_embodiment(source: Path, embodiment_source: Path) -> Path:
    temporary_value = os.environ.get("TMPDIR")
    if not temporary_value:
        raise RuntimeError("RoboTwin 2 worker requires TMPDIR")
    temporary = Path(temporary_value).resolve()
    if not temporary.is_dir():
        raise RuntimeError("RoboTwin 2 TMPDIR is absent")
    overlay = temporary / "robotwin2_embodiment" / ROBOTWIN2_EMBODIMENT[0]
    config = _embodiment_config(embodiment_source)
    for key in ("urdf_path", "srdf_path"):
        value = config.get(key)
        if not isinstance(value, str):
            raise RuntimeError(f"RoboTwin 2 embodiment {key} is invalid")
        resolved = (embodiment_source / value).resolve()
        try:
            resolved.relative_to(embodiment_source)
        except ValueError as exc:
            raise RuntimeError(f"RoboTwin 2 embodiment {key} escapes its source") from exc
        if not resolved.is_file():
            raise RuntimeError(f"RoboTwin 2 embodiment {key} is absent")
        config[key] = str(resolved)
    _write_runtime_file(
        overlay / "config.yml",
        yaml.safe_dump(config, sort_keys=False).encode("utf-8"),
    )
    for arm in ("left", "right"):
        template = embodiment_source / f"curobo_{arm}_tmp.yml"
        if not template.is_file():
            raise RuntimeError(f"RoboTwin 2 cuRobo {arm} template is absent")
        text = template.read_text(encoding="utf-8")
        if "${ASSETS_PATH}" not in text:
            raise RuntimeError(f"RoboTwin 2 cuRobo {arm} template has no asset placeholder")
        rendered = text.replace("${ASSETS_PATH}", str(source)).replace("$ASSETS_PATH", str(source))
        if "ASSETS_PATH" in rendered:
            raise RuntimeError(f"RoboTwin 2 cuRobo {arm} template remains unresolved")
        planner = yaml.safe_load(rendered)
        kinematics = planner.get("robot_cfg", {}).get("kinematics", {})
        for key in ("urdf_path", "collision_spheres"):
            value = kinematics.get(key)
            if not isinstance(value, str) or not Path(value).is_file():
                raise RuntimeError(f"RoboTwin 2 cuRobo {arm} {key} is invalid")
        _write_runtime_file(overlay / f"curobo_{arm}.yml", rendered.encode("utf-8"))
    return overlay


def _validated_source() -> Path:
    source_value = os.environ.get("ROBOT_AE_ROBOTWIN2_SOURCE")
    manifest_value = os.environ.get("ROBOT_AE_ROBOTWIN2_ASSET_MANIFEST")
    if not source_value or not manifest_value:
        raise RuntimeError("RoboTwin 2 worker requires source and asset manifest paths")
    source = Path(source_value).resolve()
    manifest = Path(manifest_value).resolve()
    if not (source / ".git").is_dir() or not (source / "envs" / "_base_task.py").is_file():
        raise RuntimeError("RoboTwin 2 source checkout is incomplete")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != ROBOTWIN2_SOURCE_COMMIT:
        raise RuntimeError(f"RoboTwin 2 source revision mismatch: {head}")
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"RoboTwin 2 source working tree is dirty: {dirty.splitlines()[0]}")
    required = (
        source / "assets" / "embodiments" / "aloha-agilex" / "config.yml",
        source / "assets" / "embodiments" / "aloha-agilex" / "curobo_left_tmp.yml",
        source / "assets" / "embodiments" / "aloha-agilex" / "curobo_right_tmp.yml",
        source / "assets" / "objects",
        source / "task_config" / "demo_clean.yml",
        source / "task_config" / "_camera_config.yml",
        source / "task_config" / "_embodiment_config.yml",
        source / "task_config" / "_eval_step_limit.yml",
    )
    if any(not path.exists() for path in required):
        raise RuntimeError("RoboTwin 2 clean-route assets are incomplete")
    validate_official_step_limits(
        yaml.safe_load((source / "task_config" / "_eval_step_limit.yml").read_text(encoding="utf-8"))
    )
    if not manifest.is_file():
        raise RuntimeError("RoboTwin 2 asset manifest is absent")
    read_and_validate_robotwin2_asset_record(
        assets=source / "assets",
        manifest=manifest,
        verify_archive_hashes=False,
    )
    return source


def _embodiment_config(path: Path) -> dict[str, Any]:
    config = path / "config.yml"
    if not config.is_file():
        raise RuntimeError(f"RoboTwin 2 embodiment config is absent: {config}")
    return yaml.safe_load(config.read_text(encoding="utf-8"))


def _require_headless_task_args(args: dict[str, Any]) -> None:
    if (
        type(args.get("render_freq")) is not int
        or args["render_freq"] != 0
        or args.get("eval_video_log") is not False
        or args.get("eval_video_save_dir") is not None
    ):
        raise RuntimeError("RoboTwin 2 route requires headless rendering")


def _canonical_pci(value: str) -> str:
    match = re.fullmatch(
        r"([0-9a-fA-F]{4}|[0-9a-fA-F]{8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])",
        value.strip(),
    )
    if match is None:
        raise RuntimeError(f"invalid PCI bus identifier: {value!r}")
    domain, bus, device, function = match.groups()
    return f"{domain[-4:].lower()}:{bus.lower()}:{device.lower()}.{function}"


def _physical_gpu_pci(render_gpu_id: int) -> str:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"],
        text=True,
    )
    matches = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError("nvidia-smi GPU PCI output is invalid")
        try:
            index = int(fields[0])
        except ValueError as exc:
            raise RuntimeError("nvidia-smi GPU index is invalid") from exc
        if index == render_gpu_id:
            matches.append(_canonical_pci(fields[1]))
    if len(matches) != 1:
        raise RuntimeError(f"physical GPU {render_gpu_id} has no unique PCI identifier")
    return matches[0]


def _validate_sapien_device(device: Any, expected_pci: str) -> None:
    if not device.is_cuda() or not device.can_render() or device.cuda_id != 0:
        raise RuntimeError("SAPIEN logical render device differs")
    if device.pci_string is None or _canonical_pci(device.pci_string) != expected_pci:
        raise RuntimeError("SAPIEN render device PCI differs from the assigned physical GPU")


def _validate_sapien_wrapper_modules(modules: dict[str, Any]) -> None:
    if set(modules) != set(SAPIEN_WRAPPER_SHA256):
        raise RuntimeError("SAPIEN wrapper module set differs")
    for name, expected_hash in SAPIEN_WRAPPER_SHA256.items():
        raw_path = Path(modules[name].__file__)
        path = raw_path.resolve()
        if raw_path.is_symlink() or not path.is_file() or _sha256(path.read_bytes()) != expected_hash:
            raise RuntimeError(f"SAPIEN wrapper source differs: {name}")


def _patch_sapien_render_device(
    sapien: Any,
    core: Any,
    engine_module: Any,
    renderer_module: Any,
    scene_module: Any,
    expected_pci: str,
) -> tuple[Any, Any]:
    if (
        sapien.Engine is not engine_module.Engine
        or core.Engine is not engine_module.Engine
        or sapien.Scene is not scene_module.Scene
        or core.Scene is not scene_module.Scene
        or sapien.SapienRenderer is not renderer_module.SapienRenderer
        or core.SapienRenderer is not renderer_module.SapienRenderer
        or renderer_module._SapienRenderer is not sapien.pysapien.render.SapienRenderer
    ):
        raise RuntimeError("SAPIEN wrapper bindings differ")
    selected_device = sapien.Device("cuda:0")
    _validate_sapien_device(selected_device, expected_pci)
    original_renderer = renderer_module.SapienRenderer
    original_create_scene = engine_module.Engine.create_scene

    class RoutedSapienRenderer:
        def __init__(self, **kwargs: object) -> None:
            if kwargs:
                raise RuntimeError("RoboTwin 2 does not permit SAPIEN renderer overrides")

        def create_material(self) -> Any:
            return renderer_module.RenderMaterial()

    def routed_create_scene(self: Any, config: Any = None) -> Any:
        if config is None:
            config = engine_module.SceneConfig()
        sapien.pysapien.physx.set_scene_config(config)
        render_system = sapien.pysapien.render.RenderSystem(selected_device)
        _validate_sapien_device(render_system.device, expected_pci)
        return scene_module.Scene([sapien.pysapien.physx.PhysxCpuSystem(), render_system])

    try:
        renderer_module.SapienRenderer = RoutedSapienRenderer
        sapien.SapienRenderer = RoutedSapienRenderer
        core.SapienRenderer = RoutedSapienRenderer
        engine_module.Engine.create_scene = routed_create_scene
        if (
            renderer_module.SapienRenderer is not RoutedSapienRenderer
            or sapien.SapienRenderer is not RoutedSapienRenderer
            or core.SapienRenderer is not RoutedSapienRenderer
            or engine_module.Engine.create_scene is not routed_create_scene
        ):
            raise RuntimeError("SAPIEN render-device patch installation failed")
    except BaseException:
        renderer_module.SapienRenderer = original_renderer
        sapien.SapienRenderer = original_renderer
        core.SapienRenderer = original_renderer
        engine_module.Engine.create_scene = original_create_scene
        raise
    return RoutedSapienRenderer, routed_create_scene


def _install_sapien_render_device(render_gpu_id: int) -> str:
    global _SAPIEN_RENDER_GPU_ID
    global _SAPIEN_RENDER_PCI
    global _SAPIEN_ROUTED_RENDERER
    global _SAPIEN_ROUTED_CREATE_SCENE
    if _SAPIEN_RENDER_GPU_ID is not None:
        if _SAPIEN_RENDER_GPU_ID != render_gpu_id or _SAPIEN_RENDER_PCI is None:
            raise RuntimeError("SAPIEN render assignment changed within one process")
        sapien = importlib.import_module("sapien")
        core = importlib.import_module("sapien.core")
        engine_module = importlib.import_module("sapien.wrapper.engine")
        renderer_module = importlib.import_module("sapien.wrapper.renderer")
        if (
            sapien.SapienRenderer is not _SAPIEN_ROUTED_RENDERER
            or core.SapienRenderer is not _SAPIEN_ROUTED_RENDERER
            or renderer_module.SapienRenderer is not _SAPIEN_ROUTED_RENDERER
            or engine_module.Engine.create_scene is not _SAPIEN_ROUTED_CREATE_SCENE
        ):
            raise RuntimeError("SAPIEN render-device patch changed within one process")
        return _SAPIEN_RENDER_PCI
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(render_gpu_id):
        raise RuntimeError("SAPIEN render assignment differs from CUDA_VISIBLE_DEVICES")
    consumers = sorted(name for name in sys.modules if name == "envs" or name.startswith("envs."))
    if consumers:
        raise RuntimeError(f"RoboTwin 2 environment modules were imported too early: {consumers}")
    sapien = importlib.import_module("sapien")
    if sapien.__version__ != SAPIEN_VERSION:
        raise RuntimeError(f"SAPIEN version differs: {sapien.__version__}")
    modules = {
        name: importlib.import_module(name)
        for name in SAPIEN_WRAPPER_SHA256
    }
    _validate_sapien_wrapper_modules(modules)
    core = importlib.import_module("sapien.core")
    expected_pci = _physical_gpu_pci(render_gpu_id)
    routed_renderer, routed_create_scene = _patch_sapien_render_device(
        sapien,
        core,
        modules["sapien.wrapper.engine"],
        modules["sapien.wrapper.renderer"],
        modules["sapien.wrapper.scene"],
        expected_pci,
    )
    _SAPIEN_RENDER_GPU_ID = render_gpu_id
    _SAPIEN_RENDER_PCI = expected_pci
    _SAPIEN_ROUTED_RENDERER = routed_renderer
    _SAPIEN_ROUTED_CREATE_SCENE = routed_create_scene
    return expected_pci


def _load_task(source: Path, task_id: str, render_gpu_id: int) -> tuple[Any, dict[str, Any]]:
    _install_sapien_render_device(render_gpu_id)
    _install_runtime_curobo_task_configs()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    envs = importlib.import_module("envs")
    if Path(envs.__file__).resolve() != source / "envs" / "__init__.py":
        raise RuntimeError("imported RoboTwin 2 package differs from the pinned source")
    module = importlib.import_module(f"envs.{task_id}")
    if Path(module.__file__).resolve() != source / "envs" / f"{task_id}.py":
        raise RuntimeError("imported RoboTwin 2 task differs from the pinned source")
    task_class = getattr(module, task_id, None)
    if not isinstance(task_class, type):
        raise RuntimeError(f"RoboTwin 2 task class is absent: {task_id}")
    args = yaml.safe_load((source / "task_config" / f"{ROBOTWIN2_CONFIG}.yml").read_text(encoding="utf-8"))
    embodiment = tuple(args.get("embodiment", ()))
    if embodiment != ROBOTWIN2_EMBODIMENT:
        raise RuntimeError(f"RoboTwin 2 embodiment differs: {embodiment}")
    embodiment_table = yaml.safe_load(
        (source / "task_config" / "_embodiment_config.yml").read_text(encoding="utf-8")
    )
    camera_table = yaml.safe_load((source / "task_config" / "_camera_config.yml").read_text(encoding="utf-8"))
    robot_source = (source / embodiment_table[embodiment[0]]["file_path"]).resolve()
    robot_path = _runtime_embodiment(source, robot_source)
    args.update(
        {
            "task_name": task_id,
            "task_config": ROBOTWIN2_CONFIG,
            "ckpt_setting": "2toINF-X-VLA-RoboTwin2-a157c580",
            "policy_name": "X-VLA",
            "left_robot_file": str(robot_path),
            "right_robot_file": str(robot_path),
            "left_embodiment_config": _embodiment_config(robot_path),
            "right_embodiment_config": _embodiment_config(robot_path),
            "dual_arm_embodied": True,
            "embodiment_name": embodiment[0],
            "head_camera_h": camera_table[args["camera"]["head_camera_type"]]["h"],
            "head_camera_w": camera_table[args["camera"]["head_camera_type"]]["w"],
            "eval_mode": True,
            "render_freq": 0,
            "eval_video_log": False,
        }
    )
    _require_headless_task_args(args)
    return task_class(), args


def _official_unseen_instruction(
    source: Path,
    task_id: str,
    episode_info: Any,
    policy_seed: int,
) -> str:
    if (
        not isinstance(episode_info, Mapping)
        or not isinstance(episode_info.get("info"), Mapping)
        or any(type(key) is not str or type(value) is not str for key, value in episode_info["info"].items())
    ):
        raise RuntimeError("RoboTwin 2 expert replay returned invalid episode information")
    generator_path = source / "description" / "utils" / "generate_episode_instructions.py"
    spec = importlib.util.spec_from_file_location("_robotwin2_generate_episode_instructions", generator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("RoboTwin 2 instruction generator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != generator_path.resolve():
        raise RuntimeError("RoboTwin 2 instruction generator differs from the pinned source")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(policy_seed)
        np.random.seed(policy_seed % (2**32))
        generated = module.generate_episode_descriptions(task_id, [dict(episode_info["info"])], 100)
        if (
            type(generated) is not list
            or len(generated) != 1
            or type(generated[0]) is not dict
            or type(generated[0].get("unseen")) is not list
            or not generated[0]["unseen"]
            or any(type(item) is not str or not item.strip() for item in generated[0]["unseen"])
        ):
            raise RuntimeError("RoboTwin 2 official unseen instructions are invalid")
        instruction = str(np.random.choice(generated[0]["unseen"]))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    if not instruction.strip():
        raise RuntimeError("RoboTwin 2 official unseen instruction is empty")
    return instruction


class RoboTwin2Worker:
    ACTION_SPEC = ROBOTWIN_ACTION_SPEC

    def __init__(self, profile: Profile, episode: EpisodeKey, *, render_gpu_id: int) -> None:
        if not isinstance(profile, Profile) or not isinstance(episode, EpisodeKey):
            raise StrictSchemaError("RoboTwin 2 worker requires Profile and EpisodeKey")
        if profile.environment.suite != "robotwin2_demo_clean":
            raise StrictSchemaError("RoboTwin 2 worker requires the demo_clean suite")
        if profile.policy.action_spec != self.ACTION_SPEC:
            raise StrictSchemaError("RoboTwin 2 worker action spec differs from profile")
        if profile.policy.chunk_horizon != 1 or profile.policy.execution_count != 1:
            raise StrictSchemaError("RoboTwin 2 worker requires one-action policy responses")
        if episode.task_id not in ROBOTWIN_TASKS:
            raise StrictSchemaError("RoboTwin 2 worker received an unknown task")
        related_protocols = {ROBOTWIN2_RELATED_PROTOCOL, ROBOTWIN2_RELATED_SMOKE_PROTOCOL}
        benchmark_protocols = {ROBOTWIN2_BENCHMARK_PROTOCOL, ROBOTWIN2_BENCHMARK_SMOKE_PROTOCOL}
        related_tasks = {task for pair in ROBOTWIN2_RELATED_PAIRS for task in pair}
        if episode.protocol in related_protocols:
            if episode.split not in {"evolve", "selection", "transfer"} or episode.task_id not in related_tasks:
                raise StrictSchemaError("RoboTwin 2 related-transfer episode differs")
        elif episode.protocol in benchmark_protocols:
            if episode.split != "benchmark":
                raise StrictSchemaError("RoboTwin 2 benchmark episode differs")
        else:
            raise StrictSchemaError("RoboTwin 2 episode protocol differs")
        if (
            episode.scenario_id != ROBOTWIN2_SCENARIO
            or episode.horizon != expected_horizon(episode.protocol, episode.task_id)
        ):
            raise StrictSchemaError("RoboTwin 2 episode scenario or horizon differs")
        if type(render_gpu_id) is not int or render_gpu_id < 0:
            raise StrictSchemaError("render_gpu_id must be a nonnegative int")
        self._profile = profile
        self._episode = episode
        self._render_gpu_id = render_gpu_id
        self._env: Any = None
        self._observation: dict[str, Any] | None = None
        self._instruction: str | None = None
        self._step = 0
        self._success = False
        self._closed = False

    def reset(self) -> None:
        if self._closed or self._env is not None:
            raise RuntimeError("RoboTwin 2 worker reset is single-use")
        if os.environ.get("PYTHONNOUSERSITE") != "1":
            raise RuntimeError("RoboTwin 2 worker requires PYTHONNOUSERSITE=1")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != str(self._render_gpu_id):
            raise RuntimeError("RoboTwin 2 CUDA and render assignment differ")
        source = _validated_source()
        os.chdir(source)
        environment, args = _load_task(source, self._episode.task_id, self._render_gpu_id)
        episode_info: Any = None
        expert_success = False
        try:
            environment.setup_demo(now_ep_num=0, seed=self._episode.environment_seed, is_test=True, **args)
            episode_info = environment.play_once()
            expert_success = bool(environment.plan_success and environment.check_success())
        finally:
            environment.close_env()
        if not expert_success:
            raise RuntimeError("RoboTwin 2 episode seed failed the official expert-solvability check")
        instruction = _official_unseen_instruction(
            source,
            self._episode.task_id,
            episode_info,
            self._episode.policy_seed,
        )
        try:
            environment.setup_demo(now_ep_num=0, seed=self._episode.environment_seed, is_test=True, **args)
            environment.set_instruction(instruction=instruction)
            observation = environment.get_obs()
        except BaseException:
            environment.close_env()
            raise
        self._instruction = instruction
        self._env = environment
        self._observation = observation
        self._step = 0
        self._success = False

    def observe(self) -> FairObservation:
        if self._env is None or self._observation is None or self._instruction is None or self._closed:
            raise RuntimeError("RoboTwin 2 worker is not active")
        raw = self._observation
        camera_specs = {item.name: item for item in self._profile.environment.cameras}
        image_keys = {"head": "head_camera", "left_wrist": "left_camera", "right_wrist": "right_camera"}
        if set(camera_specs) != set(image_keys):
            raise StrictSchemaError("RoboTwin 2 profile cameras differ from the official X-VLA client")
        cameras = {}
        for name, raw_name in image_keys.items():
            image = np.ascontiguousarray(raw["observation"][raw_name]["rgb"], dtype=np.uint8)
            expected = camera_specs[name]
            if image.shape != (expected.height, expected.width, 3):
                raise RuntimeError(f"RoboTwin 2 camera {name!r} has invalid shape {image.shape}")
            cameras[name] = CameraObservation(
                frame_id=expected.frame_id,
                optical_convention=expected.optical_convention,
                rgb=image,
                depth_m=None,
                depth_valid=None,
                intrinsics=None,
                camera_to_world=None,
            )
        state_values = {
            "left_eef_pose": np.asarray(raw["endpose"]["left_endpose"], dtype=np.float32),
            "left_gripper_position": np.asarray((raw["endpose"]["left_gripper"],), dtype=np.float32),
            "right_eef_pose": np.asarray(raw["endpose"]["right_endpose"], dtype=np.float32),
            "right_gripper_position": np.asarray((raw["endpose"]["right_gripper"],), dtype=np.float32),
        }
        vectors = tuple(
            RobotStateVector(spec, np.ascontiguousarray(state_values[spec.name], dtype=np.float32))
            for spec in self._profile.environment.robot_state
        )
        return FairObservation(
            episode_id=self._episode.artifact_id(),
            step_index=self._step,
            timestamp_ns=self._step,
            instruction=self._instruction,
            cameras=cameras,
            proprioception=RobotProprioception(vectors),
        )

    def apply(self, action: CanonicalActionChunk) -> None:
        if self._env is None or self._closed:
            raise RuntimeError("RoboTwin 2 worker is not active")
        if not isinstance(action, CanonicalActionChunk) or action.spec != self.ACTION_SPEC:
            raise StrictSchemaError("RoboTwin 2 action spec mismatch")
        if action.execution_count != 1 or action.horizon != 1 or action.start_step != self._step:
            raise StrictSchemaError("RoboTwin 2 worker requires one action at the current step")
        native = np.asarray(action.executable_values()[0], dtype=np.float64)
        if native.shape != (16,):
            raise StrictSchemaError("RoboTwin 2 native action must have width 16")
        if native[7] not in {-1.0, 1.0} or native[15] not in {-1.0, 1.0}:
            raise StrictSchemaError("RoboTwin 2 gripper actions must be -1 or 1")
        if not np.isclose(np.linalg.norm(native[3:7]), 1.0, atol=1e-4):
            raise StrictSchemaError("RoboTwin 2 left quaternion must be normalized")
        if not np.isclose(np.linalg.norm(native[11:15]), 1.0, atol=1e-4):
            raise StrictSchemaError("RoboTwin 2 right quaternion must be normalized")
        self._env.take_action(native, action_type="ee")
        self._observation = self._env.get_obs()
        self._observation["endpose"]["left_endpose"] = native[:7].tolist()
        self._observation["endpose"]["right_endpose"] = native[8:15].tolist()
        self._success = self._success or bool(self._env.check_success())
        self._step += 1

    def private_success(self) -> bool:
        if self._env is None or self._closed:
            raise RuntimeError("RoboTwin 2 worker is not active")
        return self._success

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        environment = self._env
        self._env = None
        if environment is not None:
            environment.close_env()
        self._observation = None
