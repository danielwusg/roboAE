from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from robot_auto_evolve.agent.framing import read_frame, write_frame
from robot_auto_evolve.benchmarks.calvin_worker import calvin_egl_environment
from robot_auto_evolve.benchmarks.libero_paths import write_libero_config
from robot_auto_evolve.benchmarks.libero_pro_paths import default_asset_root, write_libero_pro_config
from robot_auto_evolve.benchmarks.robocerebra_paths import write_robocerebra_config
from robot_auto_evolve.benchmarks.openvla_simpler_worker import is_openvla_simpler_adapter, validate_openvla_simpler_source
from robot_auto_evolve.benchmarks.simpler_worker import validate_simpler_source
from robot_auto_evolve.config import Profile
from robot_auto_evolve.process_lifecycle import register_owned_process, unregister_owned_process
from robot_auto_evolve.protocol import CanonicalActionChunk, FairObservation, StrictSchemaError
from robot_auto_evolve.provenance import EpisodeKey
from robot_auto_evolve.runtime_paths import RuntimePaths, assert_run_runtime_path, clean_import_environment, clean_python_path, project_root_from_package
from robot_auto_evolve.services.supervisor import scrubbed_service_environment

from .private_metrics import validate_private_metrics


class SimulatorProcessError(RuntimeError):
    pass


ROBOTWIN2_MAX_TMPDIR_BYTES = 120


class SimulatorProcess:
    def __init__(
        self,
        simulator_python: Path,
        profile: Profile,
        episode: EpisodeKey,
        *,
        physical_gpu_id: int,
        runtime_dir: Path,
        source_root: Path | None = None,
        start_timeout_s: float = 60.0,
        call_timeout_s: float = 120.0,
    ) -> None:
        python = Path(simulator_python).resolve()
        if not python.is_file():
            raise StrictSchemaError("simulator.python: expected conda Python")
        if not isinstance(profile, Profile) or not isinstance(episode, EpisodeKey):
            raise StrictSchemaError("simulator: expected Profile and EpisodeKey")
        if type(physical_gpu_id) is not int or physical_gpu_id < 0:
            raise StrictSchemaError("simulator.physical_gpu_id: expected nonnegative int")
        self.python = python
        self.profile = profile
        self.episode = episode
        self.physical_gpu_id = physical_gpu_id
        self.runtime_dir = Path(runtime_dir).resolve()
        self.source_root = None if source_root is None else Path(source_root).resolve()
        self.project_root = project_root_from_package()
        self.runtime_paths = RuntimePaths.load(self.project_root)
        assert_run_runtime_path(self.project_root, self.runtime_dir)
        self.start_timeout_s = float(start_timeout_s)
        self.call_timeout_s = float(call_timeout_s)
        self.process: subprocess.Popen[bytes] | None = None
        self._stderr: Any = None
        self._sequence = 0
        self._scratch_dir: Path | None = None
        self._scratch_record: Path | None = None

    def _write_scratch_record(self, state: str) -> None:
        scratch = self._scratch_dir
        record = self._scratch_record
        if scratch is None or record is None:
            return
        payload = {
            "schema_version": 1,
            "kind": "robotwin2_short_scratch",
            "state": state,
            "path": scratch.relative_to(self.project_root).as_posix(),
            "runtime_path_sha256": hashlib.sha256(os.fsencode(str(self.runtime_dir))).hexdigest(),
            "tmpdir_bytes": len(os.fsencode(str(scratch / "tmp"))),
        }
        temporary = record.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, record)

    def _activate_robotwin2_scratch(self) -> tuple[Path, Path, Path, Path]:
        if self.source_root is None:
            raise SimulatorProcessError("RoboTwin 2 simulator requires a pinned source root")
        runs = self.project_root / "runs"
        if not self.runtime_dir.is_relative_to(runs):
            raise SimulatorProcessError("RoboTwin 2 runtime must stay below project runs")
        relative = self.runtime_dir.relative_to(runs)
        if len(relative.parts) < 2:
            raise SimulatorProcessError("RoboTwin 2 runtime must identify a run")
        parent = runs / relative.parts[0] / "runtime" / ".rt2"
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise SimulatorProcessError("RoboTwin 2 scratch parent is invalid")
        prefix = hashlib.sha256(os.fsencode(str(self.runtime_dir))).hexdigest()[:12] + "-"
        allocated = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
        try:
            scratch = allocated.resolve()
            self._scratch_dir = scratch
            self._scratch_record = self.runtime_dir / "scratch.json"
            if scratch.parent != parent.resolve() or scratch.is_symlink():
                raise SimulatorProcessError("RoboTwin 2 scratch allocation escaped its parent")
            home = scratch / "home"
            temporary = scratch / "tmp"
            cache = scratch / "cache"
            matplotlib = scratch / "matplotlib"
            for path in (home, temporary, cache, matplotlib):
                path.mkdir()
            if len(os.fsencode(str(temporary))) > ROBOTWIN2_MAX_TMPDIR_BYTES:
                raise SimulatorProcessError("RoboTwin 2 project path is too long for NVRTC scratch")
            self._write_scratch_record("active")
            return home, temporary, cache, matplotlib
        except BaseException:
            try:
                shutil.rmtree(allocated)
            except FileNotFoundError:
                pass
            self._scratch_dir = None
            self._scratch_record = None
            raise

    def _cleanup_robotwin2_scratch(self) -> None:
        scratch = self._scratch_dir
        if scratch is None:
            return
        try:
            for root, directories, _ in os.walk(scratch):
                root_path = Path(root)
                root_path.chmod(stat.S_IMODE(root_path.stat().st_mode) | stat.S_IWUSR | stat.S_IXUSR)
                for name in directories:
                    path = root_path / name
                    if not path.is_symlink():
                        path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR | stat.S_IXUSR)
            shutil.rmtree(scratch)
        except FileNotFoundError:
            pass
        self._write_scratch_record("cleaned")
        self._scratch_dir = None

    def _rpc(self, operation: str, payload: Mapping[str, Any], timeout_s: float | None = None) -> Any:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise SimulatorProcessError("simulator process is not started")
        if process.poll() is not None:
            raise SimulatorProcessError(f"simulator process exited with code {process.returncode}")
        self._sequence += 1
        sequence = self._sequence
        write_frame(process.stdin.fileno(), {"sequence": sequence, "operation": operation, "payload": dict(payload)})
        response = read_frame(process.stdout.fileno(), self.call_timeout_s if timeout_s is None else timeout_s)
        if not isinstance(response, Mapping) or set(response) != {"sequence", "ok", "result", "error"}:
            raise SimulatorProcessError("simulator returned an invalid envelope")
        if response["sequence"] != sequence:
            raise SimulatorProcessError("simulator response sequence mismatch")
        if response["ok"] is not True:
            raise SimulatorProcessError(str(response["error"]))
        return response["result"]

    def start(self) -> None:
        if self.process is not None:
            raise SimulatorProcessError("simulator process is already started")
        self.runtime_dir.mkdir(parents=True, exist_ok=False)
        home = self.runtime_dir / "home"
        temporary = self.runtime_dir / "tmp"
        if self.profile.environment.suite != "robotwin2_demo_clean":
            home.mkdir()
            temporary.mkdir()
        overlay = {
            "CONDA_PREFIX": str(self.python.parent.parent),
            "CUDA_VISIBLE_DEVICES": str(self.physical_gpu_id),
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(self.physical_gpu_id),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(home),
            "MPLCONFIGDIR": str(self.runtime_dir / "matplotlib"),
            "PATH": os.pathsep.join((str(self.python.parent), "/usr/bin", "/bin")),
            "ROBOSUITE_LOG_PATH": str(self.runtime_dir / "robosuite.log"),
            "TMPDIR": str(temporary),
            "XDG_CACHE_HOME": str(self.runtime_dir / "cache"),
            **clean_import_environment(self.project_root, self.runtime_paths),
        }
        # Test-only: thread the smoke-horizon cap into the scrubbed worker env so the
        # worker's _validate_episode skips its strict episode.horizon==catalog check.
        cap = os.environ.get("ROBOT_AE_SMOKE_HORIZON")
        if cap:
            overlay["ROBOT_AE_SMOKE_HORIZON"] = cap
        if self.profile.environment.suite == "robocerebra_public60":
            if self.source_root is None:
                raise SimulatorProcessError("RoboCerebra simulator requires a pinned source root")
            asset_root = self.runtime_paths.artifact("robocerebra_assets")
            asset_manifest = self.project_root / "manifests" / "robocerebra_assets.json"
            asset_lock = self.runtime_paths.artifact("robocerebra_asset_lock")
            case_catalog = self.project_root / "manifests" / "robocerebra_cases.json"
            required = (
                self.source_root / "LIBERO" / "libero" / "libero" / "envs" / "bddl_base_domain.py",
                asset_root,
                asset_manifest,
                asset_lock,
                case_catalog,
            )
            if any(not path.exists() for path in required):
                raise SimulatorProcessError("RoboCerebra simulator sources or assets are incomplete")
            config = self.runtime_dir / "robocerebra_libero_config"
            try:
                write_robocerebra_config(self.source_root, config)
            except RuntimeError as exc:
                raise SimulatorProcessError(str(exc)) from exc
            overlay.update(
                {
                    "LIBERO_CONFIG_PATH": str(config),
                    "ROBOT_AE_ROBOCEREBRA_SOURCE": str(self.source_root),
                    "ROBOT_AE_ROBOCEREBRA_ASSETS": str(asset_root),
                    "ROBOT_AE_ROBOCEREBRA_ASSET_MANIFEST": str(asset_manifest),
                    "ROBOT_AE_ROBOCEREBRA_ASSET_LOCK": str(asset_lock),
                    "ROBOT_AE_ROBOCEREBRA_CASE_CATALOG": str(case_catalog),
                    "PYTHONPATH": clean_python_path(self.project_root, self.source_root / "LIBERO"),
                }
            )
        elif self.profile.environment.suite.startswith("libero_pro_"):
            if self.source_root is None:
                raise SimulatorProcessError("LIBERO-Pro simulator requires a pinned source root")
            assets = default_asset_root(self.project_root)
            config = self.runtime_dir / "libero_pro_config"
            try:
                write_libero_pro_config(self.source_root, assets, config)
            except RuntimeError as exc:
                raise SimulatorProcessError(str(exc)) from exc
            overlay["LIBERO_CONFIG_PATH"] = str(config)
            overlay["ROBOT_AE_LIBERO_PRO_SOURCE"] = str(self.source_root)
            overlay["ROBOT_AE_LIBERO_PRO_ASSETS"] = str(assets)
            overlay["PYTHONPATH"] = clean_python_path(self.project_root, self.source_root)
        elif self.profile.environment.suite.startswith("libero"):
            if self.source_root is None:
                raise SimulatorProcessError("LIBERO simulator requires a pinned source root")
            config = self.runtime_dir / "libero_config"
            try:
                write_libero_config(self.source_root, config)
            except RuntimeError as exc:
                raise SimulatorProcessError(str(exc)) from exc
            overlay["LIBERO_CONFIG_PATH"] = str(config)
            overlay["ROBOT_AE_LIBERO_SOURCE"] = str(self.source_root)
            overlay["PYTHONPATH"] = clean_python_path(self.project_root, self.source_root)
        elif self.profile.environment.suite.startswith("calvin"):
            if self.source_root is None:
                raise SimulatorProcessError("CALVIN simulator requires a pinned source root")
            calvin_env = self.source_root / "calvin_env"
            xvla_source = self.runtime_paths.source("X-VLA")
            sequence_manifest = self.project_root / "manifests" / "calvin_official_sequences.json"
            required = (
                calvin_env / "calvin_env" / "envs" / "play_table_env.py",
                xvla_source / "evaluation" / "calvin" / "ABC_D" / "validation" / ".hydra" / "merged_config.yaml",
                sequence_manifest,
            )
            if any(not path.is_file() for path in required):
                raise SimulatorProcessError("pinned CALVIN simulator sources are incomplete")
            overlay.update(
                {
                    "ROBOT_AE_CALVIN_SEQUENCE_MANIFEST": str(sequence_manifest),
                    "ROBOT_AE_CALVIN_SOURCE": str(self.source_root),
                    "ROBOT_AE_XVLA_SOURCE": str(xvla_source),
                }
            )
            overlay.update(calvin_egl_environment(self.physical_gpu_id))
            overlay["PYTHONPATH"] = clean_python_path(self.project_root, calvin_env)
        elif self.profile.environment.suite.startswith("simpler_"):
            if self.source_root is None:
                raise SimulatorProcessError("SimplerEnv simulator requires a prepared source root")
            try:
                openvla_route = is_openvla_simpler_adapter(self.profile.environment.adapter)
                source = (
                    validate_openvla_simpler_source(self.source_root)
                    if openvla_route
                    else validate_simpler_source(self.source_root)
                )
            except RuntimeError as exc:
                raise SimulatorProcessError(str(exc)) from exc
            icd = Path(os.environ.get("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")).resolve()
            if not icd.is_file():
                raise SimulatorProcessError("SimplerEnv simulator requires the NVIDIA Vulkan ICD")
            overlay.update({
                "ROBOT_AE_SIMPLER_OPENVLA_SOURCE": str(source) if openvla_route else "",
                "ROBOT_AE_SIMPLER_SOURCE": "" if openvla_route else str(source),
                "VK_ICD_FILENAMES": str(icd),
                "PYTHONPATH": clean_python_path(self.project_root, source, source / "ManiSkill2_real2sim"),
            })
        elif self.profile.environment.suite == "robotwin2_demo_clean":
            if self.source_root is None:
                raise SimulatorProcessError("RoboTwin 2 simulator requires a pinned source root")
            asset_manifest = self.project_root / "manifests" / "robotwin2_assets.json"
            required = (
                self.source_root / "envs" / "_base_task.py",
                self.source_root / "task_config" / "demo_clean.yml",
                asset_manifest,
            )
            if any(not path.is_file() for path in required):
                raise SimulatorProcessError("RoboTwin 2 simulator source or asset manifest is incomplete")
            icd = Path(os.environ.get("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")).resolve()
            if not icd.is_file():
                raise SimulatorProcessError("RoboTwin 2 simulator requires the NVIDIA Vulkan ICD")
            overlay.update(
                {
                    "ROBOT_AE_ROBOTWIN2_SOURCE": str(self.source_root),
                    "ROBOT_AE_ROBOTWIN2_ASSET_MANIFEST": str(asset_manifest),
                    "VK_ICD_FILENAMES": str(icd),
                    "PYTHONPATH": clean_python_path(self.project_root, self.source_root),
                }
            )
        elif self.profile.environment.suite == "vlabench_xvla_tracks_1_4":
            if self.source_root is None:
                raise SimulatorProcessError("VLABench simulator requires a pinned source root")
            package = self.source_root / "VLABench"
            asset_manifest = self.project_root / "manifests" / "vlabench_assets.json"
            required = (
                package / "envs" / "dm_env.py",
                package / "configs" / "task_config.json",
                package / "configs" / "robot_config.json",
                asset_manifest,
            )
            if any(not path.is_file() for path in required):
                raise SimulatorProcessError("VLABench simulator source or asset manifest is incomplete")
            overlay.update(
                {
                    "PYOPENGL_PLATFORM": "egl",
                    "ROBOT_AE_VLABENCH_SOURCE": str(self.source_root),
                    "ROBOT_AE_VLABENCH_ASSET_MANIFEST": str(asset_manifest),
                    "VLABENCH_ROOT": str(package),
                    "PYTHONPATH": clean_python_path(self.project_root, self.source_root),
                }
            )
        elif self.profile.environment.suite == "robocasa365_target":
            if self.source_root is None:
                raise SimulatorProcessError("RoboCasa365 simulator requires a pinned source root")
            robosuite_source = self.runtime_paths.source("robosuite")
            asset_lock = self.runtime_paths.artifact("robocasa365_asset_lock")
            required = (
                self.source_root / "robocasa" / "wrappers" / "gym_wrapper.py",
                robosuite_source / "robosuite" / "controllers" / "composite" / "composite_controller.py",
                asset_lock,
            )
            if any(not path.is_file() for path in required):
                raise SimulatorProcessError("RoboCasa365 simulator sources or asset lock are incomplete")
            overlay.update(
                {
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                    "MUJOCO_EGL_DEVICE_ID": str(self.physical_gpu_id),
                    "ROBOT_AE_ROBOCASA_SOURCE": str(self.source_root),
                    "ROBOT_AE_ROBOSUITE_SOURCE": str(robosuite_source),
                    "ROBOT_AE_ROBOCASA_ASSET_LOCK": str(asset_lock),
                    "PYTHONPATH": clean_python_path(self.project_root),
                }
            )
        elif self.profile.environment.suite == "fixture":
            overlay["PYTHONPATH"] = clean_python_path(self.project_root)
        else:
            raise SimulatorProcessError(f"unsupported simulator suite {self.profile.environment.suite!r}")
        try:
            if self.profile.environment.suite == "robotwin2_demo_clean":
                home, temporary, cache, matplotlib = self._activate_robotwin2_scratch()
                overlay.update(
                    {
                        "HOME": str(home),
                        "MPLCONFIGDIR": str(matplotlib),
                        "TMPDIR": str(temporary),
                        "XDG_CACHE_HOME": str(cache),
                    }
                )
            environment = scrubbed_service_environment(os.environ, overlay)
            self._stderr = (self.runtime_dir / "simulator.stderr.log").open("ab", buffering=0)
            self.process = subprocess.Popen(
                [str(self.python), "-m", "robot_auto_evolve.benchmarks.process_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                env=environment,
                start_new_session=True,
            )
            register_owned_process(self.process, f"simulator:{self.episode.artifact_id()}")
            result = self._rpc(
                "initialize",
                {
                    "profile": self.profile.to_mapping(),
                    "episode": self.episode.to_mapping(),
                    "render_gpu_id": self.physical_gpu_id,
                },
                self.start_timeout_s,
            )
            if result != {"ready": True}:
                raise SimulatorProcessError("simulator returned an invalid handshake")
        except Exception:
            self.close(force=True)
            raise

    def reinitialize(self, episode: EpisodeKey) -> None:
        """W3-C3: reuse this already-started subprocess for a new episode (fresh env, same
        suite/profile/render GPU). The heavy sim import stays loaded; the family worker + env
        are rebuilt from scratch, so per-episode env build/seed/reset semantics are unchanged."""
        if not isinstance(episode, EpisodeKey):
            raise StrictSchemaError("simulator.reinitialize: expected EpisodeKey")
        self.episode = episode
        result = self._rpc(
            "reinitialize",
            {
                "profile": self.profile.to_mapping(),
                "episode": episode.to_mapping(),
                "render_gpu_id": self.physical_gpu_id,
            },
            self.start_timeout_s,
        )
        if result != {"ready": True}:
            raise SimulatorProcessError("simulator returned an invalid reinitialize handshake")

    def reset(self) -> None:
        if self._rpc("reset", {}) != {"reset": True}:
            raise SimulatorProcessError("simulator returned an invalid reset response")

    def observe(self) -> FairObservation:
        return FairObservation.from_mapping(self._rpc("observe", {}))

    def apply(self, action: CanonicalActionChunk) -> None:
        if self._rpc("apply", {"action": action.to_mapping()}) != {"applied": True}:
            raise SimulatorProcessError("simulator returned an invalid apply response")

    def private_success(self) -> bool:
        result = self._rpc("private_success", {})
        if not isinstance(result, Mapping) or set(result) != {"success"} or type(result["success"]) is not bool:
            raise SimulatorProcessError("simulator returned an invalid success response")
        return result["success"]

    def private_metrics(self) -> dict[str, bool | float] | None:
        result = self._rpc("private_metrics", {})
        if not isinstance(result, Mapping) or set(result) != {"available", "metrics"}:
            raise SimulatorProcessError("simulator returned an invalid private metrics response")
        if type(result["available"]) is not bool or not isinstance(result["metrics"], Mapping):
            raise SimulatorProcessError("simulator returned an invalid private metrics response")
        if not result["available"]:
            if result["metrics"] != {}:
                raise SimulatorProcessError("simulator returned unavailable private metrics with values")
            return None
        try:
            return validate_private_metrics(result["metrics"], "simulator.private_metrics")
        except StrictSchemaError as exc:
            raise SimulatorProcessError(str(exc)) from exc

    def runtime_info(self) -> dict[str, Any]:
        result = self._rpc("runtime_info", {})
        if not isinstance(result, Mapping):
            raise SimulatorProcessError("simulator returned invalid runtime information")
        return dict(result)

    def close(self, force: bool = False) -> None:
        process = self.process
        try:
            if process is not None:
                graceful_close = False
                if not force and process.poll() is None:
                    try:
                        result = self._rpc("close", {}, 5.0)
                        if result != {"closed": True}:
                            raise SimulatorProcessError("simulator returned an invalid close response")
                        graceful_close = True
                    except Exception:
                        force = True
                if graceful_close:
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        pass
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=5.0)
                    except (ProcessLookupError, subprocess.TimeoutExpired):
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5.0)
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
                unregister_owned_process(process)
                self.process = None
        finally:
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None
            self._cleanup_robotwin2_scratch()

    def __enter__(self) -> "SimulatorProcess":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(force=exc is not None)
