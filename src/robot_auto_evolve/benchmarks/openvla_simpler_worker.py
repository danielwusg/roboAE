from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from robot_auto_evolve.benchmarks.openvla import OPENVLA_GOOGLE_ACTION_SPEC
from robot_auto_evolve.protocol import StrictSchemaError

from .simpler_worker import _SimplerWorker, _google_task, google_scenario_grid
from .smoke_horizon import smoke_horizon_override


SOURCE_COMMIT = "ccfe3809766839a2fcfb7a3d3c9abff585189188"
SUBMODULE_COMMIT = "cd45dd27dc6bb26d048cb6570cdab4e3f935cc37"
CONTROL_MODE = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
CATALOG_SHA256 = "03974dc5e358c55eb3be374f21862b9581c0921022b0fab944e18aa7fcac1bf0"
CRITICAL_HASHES = {
    "simpler_env/__init__.py": "4e6cc2ef695a58ad552f2139d3d49be683d4da6441d74691157120f928f75800",
    "simpler_env/evaluation/maniskill2_evaluator.py": "636eee9d108636bc68f970625dad5776c58ddbe0715d379dc7aacfccce6177a8",
    "simpler_env/policies/openvla/openvla_model.py": "74da205be0de0c86b4219d99393dc92fbf0e92fc2190bd0144ae4ce6c30cdc7b",
    "simpler_env/utils/env/env_builder.py": "2d5710df48917bf27c533463d9e5110c2ea209665d22ace67e6cdfd7b93eb005",
    "simpler_env/utils/env/observation_utils.py": "7ff0950e7055fa7f9c248ccf78a38b6943a8ebf1f3542a8d484dd17e437a3cb5",
    "ManiSkill2_real2sim/mani_skill2_real2sim/agents/configs/google_robot/defaults.py": "ecef6b2ba76b7925dfdf8c02ae328c36d1dabd897e1990266a52dd61e99973ce",
    "ManiSkill2_real2sim/mani_skill2_real2sim/agents/controllers/pd_ee_pose.py": "b3d62a3f46ae9eb69f017b91671d8d29de818a51e5aa7ac9066628fabdcf3538",
    "ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/base_env.py": "532cf7981ae3c13c292f3bac913e2d17baa1b5f8316d437da0c0674b07d9a77b",
    "ManiSkill2_real2sim/mani_skill2_real2sim/utils/wrappers/observation.py": "7a7ca714b927dff18aafe984210e1e8f31d55b7aafa166e235189922b978f22b",
}
SCRIPT_HASHES = {
    "scripts/drawer_variant_agg.sh": "a6fa4e1f82be9cdb3911f5a372222dfa8d4276c200bc72c7dd736993fb8db905",
    "scripts/drawer_visual_matching.sh": "7951d13b4db5bbd4f7a27b2f1a529c16b0f8d2e9cf2113c1481530f3e54afa09",
    "scripts/move_near_variant_agg.sh": "93f1a76b706c7c10de694043aa7ba704d42004c0e52756a9f77660ea857db544",
    "scripts/move_near_visual_matching.sh": "f922f7a64457a59b231d8a360ceb291e33a4426899a52f849cdf27f4cad94c51",
    "scripts/pick_coke_can_variant_agg.sh": "a653efb2a0e2796a6491933c6933f286373b130e7a70f73f103b477acae3a967",
    "scripts/pick_coke_can_visual_matching.sh": "46127dcf0e2d0a2ad82fa246284a5bde5aafabefde4dd68ed8ca2bfee237def7",
    "scripts/put_in_drawer_variant_agg.sh": "806751bf3fd1019ed47067a758f1658ebc8cedb388af94fb8c67a3110aca5df0",
    "scripts/put_in_drawer_visual_matching.sh": "bf9ee53b98e8eb910e508425bca1300a3bdeee6096cc2081b9071c610d0e11ea",
}
PROTOCOLS = {
    "va": frozenset(
        {
            "openvla_google_va_complete_grid_v1",
            "openvla_google_va_drawer_transfer_v1",
            "openvla_google_va_full_stack_smoke_v1",
        }
    ),
    "vm": frozenset(
        {
            "openvla_google_vm_complete_grid_v1",
            "openvla_google_vm_drawer_transfer_v1",
            "openvla_google_vm_full_stack_smoke_v1",
        }
    ),
}
OPENVLA_SIMPLER_ADAPTERS = frozenset(
    {
        "robot_auto_evolve.benchmarks.openvla_simpler_worker:OpenVLASimplerGoogleVAWorker",
        "robot_auto_evolve.benchmarks.openvla_simpler_worker:OpenVLASimplerGoogleVMWorker",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_openvla_simpler_source(source: str | Path, *, full_tree: bool = False) -> Path:
    root = Path(source).resolve()
    if (root / ".robot_auto_evolve_xvla.json").exists():
        raise RuntimeError("OpenVLA SimplerEnv source cannot be the X-VLA-derived tree")
    if not (root / ".git").is_dir():
        raise RuntimeError("OpenVLA SimplerEnv source has no Git metadata")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    submodule = root / "ManiSkill2_real2sim"
    if not submodule.is_dir():
        raise RuntimeError("OpenVLA SimplerEnv simulator submodule is absent")
    submodule_head = subprocess.check_output(
        ["git", "-C", str(submodule), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != SOURCE_COMMIT or submodule_head != SUBMODULE_COMMIT:
        raise RuntimeError("OpenVLA SimplerEnv source revision differs")
    expected = {**CRITICAL_HASHES, **SCRIPT_HASHES}
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("OpenVLA SimplerEnv critical path escapes source") from exc
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"OpenVLA SimplerEnv critical file differs: {relative}")
    if full_tree:
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
            text=True,
        ).strip()
        submodule_dirty = subprocess.check_output(
            ["git", "-C", str(submodule), "status", "--porcelain=v1", "--untracked-files=all"],
            text=True,
        ).strip()
        if dirty or submodule_dirty:
            raise RuntimeError("OpenVLA SimplerEnv source tree is dirty")
    return root


def validated_openvla_simpler_source() -> Path:
    value = os.environ.get("ROBOT_AE_SIMPLER_OPENVLA_SOURCE")
    if not value:
        raise RuntimeError("OpenVLA SimplerEnv worker requires ROBOT_AE_SIMPLER_OPENVLA_SOURCE")
    return validate_openvla_simpler_source(value)


def is_openvla_simpler_adapter(value: str) -> bool:
    return value in OPENVLA_SIMPLER_ADAPTERS


def _catalog_path() -> Path:
    return Path(__file__).resolve().parents[3] / "manifests" / "openvla_simpler_scenarios.json"


def openvla_simpler_catalog(variant: str) -> tuple[dict[str, Any], ...]:
    if variant not in {"va", "vm"}:
        raise StrictSchemaError("OpenVLA SimplerEnv variant must be va or vm")
    path = _catalog_path()
    if not path.is_file() or _sha256(path) != CATALOG_SHA256:
        raise RuntimeError("OpenVLA SimplerEnv scenario catalog differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "source_repository",
        "source_commit",
        "simulator_submodule_commit",
        "control_mode",
        "reset_seed_argument",
        "environment_default_main_seed",
        "environment_first_episode_seed",
        "rotation_input",
        "action_semantics",
        "script_sha256",
        "variants",
        "task_episode_counts",
    }
    if (
        set(value) != required
        or value["schema_version"] != 1
        or value["source_commit"] != SOURCE_COMMIT
        or value["simulator_submodule_commit"] != SUBMODULE_COMMIT
        or value["control_mode"] != CONTROL_MODE
        or value["reset_seed_argument"] is not None
        or value["environment_default_main_seed"] != 2022
        or value["environment_first_episode_seed"] != 40194941
        or value["rotation_input"] != "axis_angle"
        or value["action_semantics"] != ["delta"] * 7
        or value["script_sha256"] != SCRIPT_HASHES
    ):
        raise RuntimeError("OpenVLA SimplerEnv scenario catalog contract differs")
    scenarios = value["variants"].get(variant)
    if not isinstance(scenarios, list) or not scenarios:
        raise RuntimeError("OpenVLA SimplerEnv scenario catalog variant is absent")
    identities = [(item.get("task_id"), item.get("scenario_id")) for item in scenarios if isinstance(item, dict)]
    if len(identities) != len(scenarios) or len(set(identities)) != len(scenarios):
        raise RuntimeError("OpenVLA SimplerEnv scenario identities differ")
    return tuple(copy.deepcopy(item) for item in scenarios)


def openvla_simpler_scenario_grid(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    converted = {
        "robot_init_x": config["robot_init_x_range"],
        "robot_init_y": config["robot_init_y_range"],
        "robot_init_rot_quat_center": config["robot_init_rot_quat_center"],
        "robot_init_rot_rpy_range": config["robot_init_rot_rpy_range"],
        "obj_variation_mode": config["obj_variation_mode"],
    }
    if config["obj_variation_mode"] == "xy":
        converted["obj_init_x_range"] = config["obj_init_x_range"]
        converted["obj_init_y_range"] = config["obj_init_y_range"]
    else:
        start, end = config["obj_episode_range"]
        if start != 0 or type(end) is not int or end < 1:
            raise StrictSchemaError("OpenVLA SimplerEnv object episode range differs")
        converted["episode_nums"] = end
    return google_scenario_grid(converted)


class _OpenVLASimplerGoogleWorker(_SimplerWorker):
    ACTION_SPEC = OPENVLA_GOOGLE_ACTION_SPEC
    CONTROL_MODE = CONTROL_MODE
    CONTROL_PERIOD_NS = 333_333_333
    SUITE = "simpler_google_va"
    VARIANT = "va"

    @staticmethod
    def _validated_source() -> Path:
        return validated_openvla_simpler_source()

    def _validate_episode(self) -> None:
        if self._episode.protocol not in PROTOCOLS[self.VARIANT]:
            raise StrictSchemaError("OpenVLA SimplerEnv episode protocol differs")
        match = re.fullmatch(r"([a-z0-9_]+)__grid_(\d{3})", self._episode.scenario_id)
        if match is None:
            raise StrictSchemaError("OpenVLA SimplerEnv scenario identifier differs")
        scenario_id = match.group(1)
        matches = [
            value
            for value in openvla_simpler_catalog(self.VARIANT)
            if value["task_id"] == self._episode.task_id and value["scenario_id"] == scenario_id
        ]
        if len(matches) != 1:
            raise StrictSchemaError("OpenVLA SimplerEnv scenario is absent")
        self._scenario = matches[0]
        self._grid_index = int(match.group(2))
        grid = openvla_simpler_scenario_grid(self._scenario)
        if self._grid_index >= len(grid):
            raise StrictSchemaError("OpenVLA SimplerEnv grid member is unavailable")
        expected_horizon = 11 if self._episode.protocol.endswith("_full_stack_smoke_v1") else self._scenario["max_episode_steps"]
        if smoke_horizon_override() is None and self._episode.horizon != expected_horizon:
            raise StrictSchemaError("OpenVLA SimplerEnv episode horizon differs")

    def _make_and_reset(self) -> tuple[Any, dict[str, Any]]:
        import gymnasium as gym

        if self._source is None:
            raise RuntimeError("OpenVLA SimplerEnv source is not validated")
        config = self._scenario
        if _google_task(config["env_name"]) != self._episode.task_id:
            raise StrictSchemaError("OpenVLA SimplerEnv task and environment differ")
        kwargs = {
            "obs_mode": "rgbd",
            "max_episode_steps": self._episode.horizon,
            "robot": config["robot_name"],
            "sim_freq": config["sim_freq"],
            "control_freq": config["control_freq"],
            "control_mode": self.CONTROL_MODE,
            "scene_name": config["scene_name"],
            "camera_cfgs": {"add_segmentation": True},
            **config["additional_env_build_kwargs"],
            "renderer_kwargs": self._renderer_kwargs(),
        }
        overlay = config["rgb_overlay_path"]
        if overlay is not None:
            path = (self._source / overlay).resolve()
            try:
                path.relative_to(self._source)
            except ValueError as exc:
                raise StrictSchemaError("OpenVLA SimplerEnv overlay path escapes source") from exc
            if not path.is_file():
                raise RuntimeError("OpenVLA SimplerEnv overlay asset is absent")
            kwargs["rgb_overlay_path"] = str(path)
            kwargs["rgb_overlay_cameras"] = ["overhead_camera"]
        import simpler_env  # noqa: F401

        environment = gym.make(config["env_name"], **kwargs)
        return environment, openvla_simpler_scenario_grid(config)[self._grid_index]

    def _reset_environment(self, options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._env.reset(options=options)

    def _update_success(self, terminated: bool) -> None:
        self._success = terminated

    def runtime_info(self) -> dict[str, Any]:
        value = super().runtime_info()
        main_seed = getattr(self._env.unwrapped, "_main_seed", None)
        episode_seed = getattr(self._env.unwrapped, "_episode_seed", None)
        if main_seed != 40194941 or episode_seed != 40194941:
            raise RuntimeError("OpenVLA SimplerEnv upstream default reset seed differs")
        value.update(
            {
                "action_semantics": ["delta"] * 7,
                "catalog_sha256": CATALOG_SHA256,
                "rotation_input": "axis_angle",
                "reset_seed_argument": None,
                "environment_default_main_seed": 2022,
                "environment_post_reset_main_seed": main_seed,
                "environment_episode_seed": episode_seed,
                "evaluation_stop_condition": "configured_horizon",
                "success_measurement": "final_step_terminated",
                "source_commit": SOURCE_COMMIT,
                "simulator_submodule_commit": SUBMODULE_COMMIT,
            }
        )
        return value


class OpenVLASimplerGoogleVAWorker(_OpenVLASimplerGoogleWorker):
    pass


class OpenVLASimplerGoogleVMWorker(_OpenVLASimplerGoogleWorker):
    SUITE = "simpler_google_vm"
    VARIANT = "vm"
