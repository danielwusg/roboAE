from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_auto_evolve.benchmarks.robolab120_batching import RoboLabBatch
from robot_auto_evolve.benchmarks.robolab120_rpc import (
    RoboLabActionBatch,
    RoboLabAppConfig,
    RoboLabObservationBatch,
    RoboLabPrivateStatusBatch,
    RoboLabRpcRequest,
    RoboLabRpcResponse,
    read_robolab_response,
    write_robolab_request,
)
from robot_auto_evolve.benchmarks.robolab120_worker import (
    ROBOLAB_ASSET_TREE_SHA256,
    ROBOLAB_SOURCE_COMMIT,
    validate_robolab_profile,
)
from robot_auto_evolve.config import Profile
from robot_auto_evolve.process_lifecycle import register_owned_process, unregister_owned_process
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.runtime_paths import RuntimePaths, assert_run_runtime_path, clean_import_environment, clean_python_path, project_root_from_package, verify_python_import_origin
from robot_auto_evolve.services.supervisor import scrubbed_service_environment


class RoboLabSimulatorError(RuntimeError):
    pass


class RoboLabSimulatorProcess:
    def __init__(
        self,
        simulator_python: Path,
        profile: Profile,
        *,
        physical_gpu_id: int,
        vector_batch_size: int,
        runtime_dir: Path,
        source_root: Path,
        start_timeout_s: float = 1800.0,
        call_timeout_s: float = 900.0,
    ) -> None:
        python = Path(simulator_python).resolve()
        source = Path(source_root).resolve()
        if not python.is_file():
            raise StrictSchemaError("robolab simulator Python is missing")
        if not (source / ".git").is_dir():
            raise StrictSchemaError("robolab simulator source is missing")
        if type(physical_gpu_id) is not int or physical_gpu_id < 0:
            raise StrictSchemaError("robolab simulator GPU id is invalid")
        if type(vector_batch_size) is not int or vector_batch_size < 1:
            raise StrictSchemaError("robolab simulator vector batch size is invalid")
        validate_robolab_profile(profile)
        self.python = python
        self.profile = profile
        self.physical_gpu_id = physical_gpu_id
        self.vector_batch_size = vector_batch_size
        self.runtime_dir = Path(runtime_dir).resolve()
        self.source_root = source
        self.project_root = project_root_from_package()
        self.runtime_paths = RuntimePaths.load(self.project_root)
        self.start_timeout_s = float(start_timeout_s)
        self.call_timeout_s = float(call_timeout_s)
        self.process: subprocess.Popen[bytes] | None = None
        self._stderr: Any = None
        self._sequence = 0

    def _rpc(self, operation: str, payload: Mapping[str, Any], timeout_s: float | None = None) -> Mapping[str, Any]:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise RoboLabSimulatorError("RoboLab simulator is not started")
        if process.poll() is not None:
            raise RoboLabSimulatorError(f"RoboLab simulator exited with code {process.returncode}")
        self._sequence += 1
        request = RoboLabRpcRequest(self._sequence, operation, payload)
        write_robolab_request(process.stdin.fileno(), request)
        response = read_robolab_response(
            process.stdout.fileno(),
            self.call_timeout_s if timeout_s is None else timeout_s,
        )
        response.validate_request(request)
        if not response.ok:
            raise RoboLabSimulatorError(str(response.error))
        if response.result is None:
            raise RoboLabSimulatorError("RoboLab simulator returned no result")
        return response.result

    def start(self) -> None:
        if self.process is not None:
            raise RoboLabSimulatorError("RoboLab simulator is already started")
        actual = subprocess.check_output(
            ["git", "-C", str(self.source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != ROBOLAB_SOURCE_COMMIT:
            raise RoboLabSimulatorError("RoboLab source revision differs")
        assert_run_runtime_path(self.project_root, self.runtime_dir)
        verify_python_import_origin(self.python, self.project_root, self.runtime_paths)
        self.runtime_dir.mkdir(parents=True, exist_ok=False)
        home = self.runtime_dir / "home"
        temporary = self.runtime_dir / "tmp"
        home.mkdir()
        temporary.mkdir()
        overlay = {
            "CONDA_PREFIX": str(self.python.parent.parent),
            "CUDA_VISIBLE_DEVICES": str(self.physical_gpu_id),
            "HOME": str(home),
            "PATH": os.pathsep.join((str(self.python.parent), "/usr/bin", "/bin")),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": clean_python_path(self.project_root, self.source_root),
            "TMPDIR": str(temporary),
            "XDG_CACHE_HOME": str(self.runtime_dir / "cache"),
            **clean_import_environment(self.project_root, self.runtime_paths),
        }
        overlay["PYTHONPATH"] = clean_python_path(self.project_root, self.source_root)
        environment = scrubbed_service_environment(os.environ, overlay)
        self._stderr = (self.runtime_dir / "simulator.stderr.log").open("ab", buffering=0)
        self.process = subprocess.Popen(
            [
                str(self.python),
                "-m",
                "robot_auto_evolve.benchmarks.robolab120_worker",
                "--source-root",
                str(self.source_root),
                "--runtime-root",
                str(self.runtime_dir / "runtime"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            env=environment,
            start_new_session=True,
        )
        register_owned_process(self.process, f"robolab-simulator:gpu{self.physical_gpu_id}")
        config = RoboLabAppConfig(
            static_profile=self.profile.to_mapping(),
            static_profile_sha256=self.profile.resolved_hash(),
            source_commit=ROBOLAB_SOURCE_COMMIT,
            asset_lock_sha256=ROBOLAB_ASSET_TREE_SHA256,
            simulator_gpu_id=self.physical_gpu_id,
            vector_batch_size=self.vector_batch_size,
        )
        try:
            if self._rpc("initialize_app", config.to_mapping(), self.start_timeout_s) != {"ready": True}:
                raise RoboLabSimulatorError("RoboLab simulator handshake differs")
        except Exception:
            self.close(force=True)
            raise

    def load_batch(self, batch: RoboLabBatch) -> None:
        result = self._rpc("load_batch", batch.to_mapping())
        if result != {"batch_id": batch.batch_id, "episode_ids": list(batch.episode_ids), "loaded": True}:
            raise RoboLabSimulatorError("RoboLab loaded batch differs")

    def observe_batch(self, batch: RoboLabBatch) -> RoboLabObservationBatch:
        result = RoboLabObservationBatch.from_mapping(
            self._rpc("observe_batch", {"batch_id": batch.batch_id})
        )
        result.validate_batch(batch)
        return result

    def apply_batch(self, batch: RoboLabBatch, actions: RoboLabActionBatch) -> None:
        actions.validate_batch(batch)
        result = self._rpc("apply_batch", actions.to_mapping())
        expected = {"batch_id": batch.batch_id, "episode_ids": list(batch.episode_ids), "applied": True}
        if result != expected:
            raise RoboLabSimulatorError("RoboLab applied batch differs")

    def private_status_batch(self, batch: RoboLabBatch) -> RoboLabPrivateStatusBatch:
        result = RoboLabPrivateStatusBatch.from_mapping(
            self._rpc("private_status_batch", {"batch_id": batch.batch_id})
        )
        result.validate_batch(batch)
        return result

    def finish_batch(self, batch: RoboLabBatch) -> None:
        result = self._rpc("finish_batch", {"batch_id": batch.batch_id})
        expected = {"batch_id": batch.batch_id, "episode_ids": list(batch.episode_ids), "finished": True}
        if result != expected:
            raise RoboLabSimulatorError("RoboLab finished batch differs")

    def candidate_barrier(self, barrier_id: str) -> None:
        result = self._rpc("candidate_barrier", {"barrier_id": barrier_id})
        if result != {"barrier_id": barrier_id, "ready": True}:
            raise RoboLabSimulatorError("RoboLab candidate barrier differs")

    def close(self, force: bool = False) -> None:
        process = self.process
        if process is None:
            return
        if not force and process.poll() is None:
            try:
                self._rpc("close", {}, 30.0)
            except Exception:
                force = True
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10.0)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        unregister_owned_process(process)
        self.process = None

    def __enter__(self) -> "RoboLabSimulatorProcess":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(force=exc is not None)
