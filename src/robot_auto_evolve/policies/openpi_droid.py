from __future__ import annotations

import base64
import contextlib
import dataclasses
import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from robot_auto_evolve.agent.api import VLARequest
from robot_auto_evolve.benchmarks.contracts import action_chunk
from robot_auto_evolve.benchmarks.droid import OpenPiDroidJointPositionAdapter
from robot_auto_evolve.protocol.schema import (
    StrictSchemaError,
    fields,
    integer,
    json_object,
    mapping,
    reject_json_constant,
    sequence,
    sha256,
    string,
)

from .config import PolicyServiceConfig
from .smoke import synthetic_request


SOURCE_COMMIT = "aa6420561529593114160d05e5ad155792b272f3"
INVENTORY_SCHEMA = "gcs_object_generation_md5_v1"
PALIGEMMA_REMOTE_ROOT = "gs://big_vision/paligemma_tokenizer.model"
PALIGEMMA_INVENTORY_SHA256 = "7099c18548f0dcbd32cf1e2008f250bd965722fb13063df48569c6de9575adf5"
PALIGEMMA_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
PALIGEMMA_SIZE = 4_264_023


@dataclass
class _Session:
    policy_seed: int
    task_id: str
    adapter: OpenPiDroidJointPositionAdapter
    initialized: bool = False
    call_index: int = 0
    cache: np.ndarray | None = None
    cache_index: int = 0
    pending_native: np.ndarray | None = None
    pending_step: int | None = None
    last_request_id: str | None = None
    last_response: dict[str, Any] | None = None


def _git_head(source: Path) -> str:
    return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()


def _require_clean(source: Path) -> None:
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"OpenPI source working tree is dirty: {dirty.splitlines()[0]}")


def _project_path(project_root: Path, value: Any, path: str) -> Path:
    text = string(value, path)
    relative = PurePosixPath(text)
    if relative.is_absolute() or text != relative.as_posix() or ".." in relative.parts or "." in relative.parts:
        raise RuntimeError(f"{path} must be a normalized project-relative path")
    return (project_root / Path(*relative.parts)).resolve()


def _hash_file(path: Path) -> tuple[int, str, str]:
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(block)
            sha.update(block)
            md5.update(block)
    return size, sha.hexdigest(), base64.b64encode(md5.digest()).decode("ascii")


def _relative_file(path: Any, field: str) -> str:
    value = string(path, field)
    relative = PurePosixPath(value)
    if relative.is_absolute() or value != relative.as_posix() or ".." in relative.parts or "." in relative.parts:
        raise RuntimeError(f"{field} must be a normalized relative path")
    return value


def _load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=json_object,
        parse_constant=reject_json_constant,
    )


def _validate_artifact(
    root: Path,
    manifest_path: Path,
    *,
    artifact_name: str,
    remote_root: str,
    inventory_sha256: str,
) -> None:
    if not root.is_dir() or not manifest_path.is_file():
        raise RuntimeError(f"{artifact_name} local artifact is incomplete")
    manifest = fields(
        _load_json(manifest_path),
        {"schema_version", "artifact_name", "source", "remote_inventory", "files", "content_sha256"},
        path="artifact",
    )
    if integer(manifest["schema_version"], "artifact.schema_version") != 1:
        raise RuntimeError("artifact schema version differs")
    if string(manifest["artifact_name"], "artifact.artifact_name") != artifact_name:
        raise RuntimeError("artifact name differs")
    source = fields(
        manifest["source"],
        {"remote_root", "inventory_schema", "file_count", "logical_size_bytes", "inventory_sha256"},
        path="artifact.source",
    )
    if string(source["remote_root"], "artifact.source.remote_root") != remote_root:
        raise RuntimeError("artifact remote root differs")
    if string(source["inventory_schema"], "artifact.source.inventory_schema") != INVENTORY_SCHEMA:
        raise RuntimeError("artifact inventory schema differs")
    if sha256(source["inventory_sha256"], "artifact.source.inventory_sha256") != inventory_sha256:
        raise RuntimeError("artifact inventory identity differs")
    remote: list[dict[str, Any]] = []
    for index, value in enumerate(sequence(manifest["remote_inventory"], "artifact.remote_inventory")):
        item = fields(value, {"generation", "md5", "path", "size"}, path=f"artifact.remote_inventory[{index}]")
        generation = string(item["generation"], f"artifact.remote_inventory[{index}].generation")
        if not generation.isdigit():
            raise RuntimeError("artifact generation must contain decimal digits")
        remote.append(
            {
                "generation": generation,
                "md5": string(item["md5"], f"artifact.remote_inventory[{index}].md5"),
                "path": _relative_file(item["path"], f"artifact.remote_inventory[{index}].path"),
                "size": integer(item["size"], f"artifact.remote_inventory[{index}].size", minimum=0),
            }
        )
    if remote != sorted(remote, key=lambda item: item["path"]):
        raise RuntimeError("artifact remote inventory is not path-sorted")
    if len({item["path"] for item in remote}) != len(remote):
        raise RuntimeError("artifact remote inventory contains duplicate paths")
    encoded_remote = json.dumps(remote, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(encoded_remote).hexdigest() != inventory_sha256:
        raise RuntimeError("artifact remote inventory hash differs")
    if integer(source["file_count"], "artifact.source.file_count") != len(remote):
        raise RuntimeError("artifact remote file count differs")
    if integer(source["logical_size_bytes"], "artifact.source.logical_size_bytes", minimum=0) != sum(item["size"] for item in remote):
        raise RuntimeError("artifact remote logical size differs")
    local: list[dict[str, Any]] = []
    for index, value in enumerate(sequence(manifest["files"], "artifact.files")):
        item = fields(value, {"path", "size", "sha256"}, path=f"artifact.files[{index}]")
        local.append(
            {
                "path": _relative_file(item["path"], f"artifact.files[{index}].path"),
                "size": integer(item["size"], f"artifact.files[{index}].size", minimum=0),
                "sha256": sha256(item["sha256"], f"artifact.files[{index}].sha256"),
            }
        )
    if local != sorted(local, key=lambda item: item["path"]):
        raise RuntimeError("artifact local inventory is not path-sorted")
    if len({item["path"] for item in local}) != len(local):
        raise RuntimeError("artifact local inventory contains duplicate paths")
    encoded_local = json.dumps(local, sort_keys=True, separators=(",", ":")).encode()
    if sha256(manifest["content_sha256"], "artifact.content_sha256") != hashlib.sha256(encoded_local).hexdigest():
        raise RuntimeError("artifact local inventory hash differs")
    remote_by_path = {item["path"]: item for item in remote}
    if [(item["path"], item["size"]) for item in local] != [(item["path"], item["size"]) for item in remote]:
        raise RuntimeError("artifact remote and local file inventories differ")
    manifest_resolved = manifest_path.resolve()
    actual: dict[str, Path] = {}
    for candidate in root.rglob("*"):
        if candidate.resolve() == manifest_resolved:
            continue
        if candidate.is_symlink():
            raise RuntimeError("artifact contains a non-regular entry")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise RuntimeError("artifact contains a non-regular entry")
        relative = candidate.relative_to(root).as_posix()
        actual[relative] = candidate
    if set(actual) != {item["path"] for item in local}:
        raise RuntimeError("artifact local file set differs")
    for item in local:
        size, digest, gcs_md5 = _hash_file(actual[item["path"]])
        if size != item["size"] or digest != item["sha256"]:
            raise RuntimeError(f"artifact local file differs: {item['path']}")
        if gcs_md5 != remote_by_path[item["path"]]["md5"]:
            raise RuntimeError(f"artifact GCS MD5 differs: {item['path']}")


def _validate_fast_tokenizer(snapshot: Path, expected: Mapping[str, Any], revision: str) -> None:
    if snapshot.name != revision:
        raise RuntimeError("FAST tokenizer snapshot revision differs")
    expected_paths = set(expected)
    actual_paths = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        raise RuntimeError("FAST tokenizer file set differs")
    for name, value in expected.items():
        item = fields(value, {"size_bytes", "sha256"}, path=f"fast_tokenizer.expected_files.{name}")
        size, digest, _ = _hash_file(snapshot / name)
        if size != integer(item["size_bytes"], f"fast_tokenizer.expected_files.{name}.size_bytes", minimum=0):
            raise RuntimeError(f"FAST tokenizer file size differs: {name}")
        if digest != sha256(item["sha256"], f"fast_tokenizer.expected_files.{name}.sha256"):
            raise RuntimeError(f"FAST tokenizer file hash differs: {name}")


@contextlib.contextmanager
def _local_downloads(download_module: Any, paligemma_path: Path) -> Iterator[None]:
    original: Callable[..., Any] = download_module.maybe_download

    def local_only(path: Any, *args: Any, **kwargs: Any) -> Path:
        value = str(path)
        if value == PALIGEMMA_REMOTE_ROOT:
            return paligemma_path
        if "://" in value:
            raise RuntimeError(f"OpenPI attempted a remote artifact access: {value}")
        return Path(value)

    download_module.maybe_download = local_only
    try:
        yield
    finally:
        download_module.maybe_download = original


def _seeded_noise(seed: int, call_index: int, horizon: int, action_dim: int) -> np.ndarray:
    if min(seed, call_index, horizon, action_dim) < 0 or horizon < 1 or action_dim < 1:
        raise ValueError("OpenPI noise parameters are invalid")
    digest = hashlib.sha256(f"openpi-pi05-noise-v1:{seed}:{call_index}".encode()).digest()
    generator = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))
    return np.ascontiguousarray(generator.standard_normal((horizon, action_dim), dtype=np.float32))


class OpenPiDroidJointPositionPolicyBackend:
    def __init__(self, config: PolicyServiceConfig, source_root: str | Path, device: str) -> None:
        if config.route.backend != "openpi_droid_jointpos":
            raise StrictSchemaError("OpenPI DROID backend received a different route")
        self.config = config
        source = Path(source_root).resolve()
        if source.name != "robolab_openpi" or not (source / "src" / "openpi" / "policies" / "policy.py").is_file():
            raise RuntimeError("OpenPI DROID inference source is incomplete")
        if _git_head(source) != SOURCE_COMMIT or config.route.source_commit != SOURCE_COMMIT:
            raise RuntimeError("OpenPI DROID inference source revision mismatch")
        _require_clean(source)
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("OpenPI DROID inference requires offline mode")
        if device != "cuda:0":
            raise RuntimeError("OpenPI DROID replica requires cuda:0")
        project_root = source.parent.parent
        checkpoint = _project_path(project_root, config.value["checkpoint_path"], "policy_config.checkpoint_path")
        artifact = _project_path(project_root, config.value["artifact_manifest_path"], "policy_config.artifact_manifest_path")
        paligemma = _project_path(project_root, config.value["paligemma_tokenizer_path"], "policy_config.paligemma_tokenizer_path")
        paligemma_artifact = _project_path(
            project_root,
            config.value["paligemma_tokenizer_artifact_manifest_path"],
            "policy_config.paligemma_tokenizer_artifact_manifest_path",
        )
        _validate_artifact(
            checkpoint,
            artifact,
            artifact_name=config.route.name.removeprefix("openpi_"),
            remote_root=string(mapping(config.value["checkpoint"], "policy_config.checkpoint")["id"], "policy_config.checkpoint.id"),
            inventory_sha256=config.route.revision,
        )
        _validate_artifact(
            paligemma.parent,
            paligemma_artifact,
            artifact_name="paligemma_tokenizer",
            remote_root=PALIGEMMA_REMOTE_ROOT,
            inventory_sha256=PALIGEMMA_INVENTORY_SHA256,
        )
        paligemma_size, paligemma_hash, _ = _hash_file(paligemma)
        if paligemma_size != PALIGEMMA_SIZE or paligemma_hash != PALIGEMMA_SHA256:
            raise RuntimeError("PaliGemma tokenizer identity differs")
        package_root = source / "src"
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        train_module = importlib.import_module("openpi.training.config")
        policy_module = importlib.import_module("openpi.policies.policy_config")
        download_module = importlib.import_module("openpi.shared.download")
        jax = importlib.import_module("jax")
        expected_train = source / "src" / "openpi" / "training" / "config.py"
        if Path(train_module.__file__).resolve() != expected_train.resolve():
            raise RuntimeError("OpenPI imported inference source differs from the pinned checkout")
        devices = tuple(jax.local_devices())
        if len(devices) != 1 or getattr(devices[0], "platform", None) != "gpu":
            raise RuntimeError("OpenPI DROID replica requires exactly one visible JAX GPU")
        train_config = train_module.get_config(string(config.value["upstream_config"], "policy_config.upstream_config"))
        fast_snapshot: Path | None = None
        if config.route.name == "openpi_pi0_fast_droid_jointpos":
            tokenizer = mapping(config.value["fast_tokenizer"], "policy_config.fast_tokenizer")
            expected_files = mapping(tokenizer["expected_files"], "policy_config.fast_tokenizer.expected_files")
            hub = importlib.import_module("huggingface_hub")
            fast_snapshot = Path(
                hub.snapshot_download(
                    repo_id=string(tokenizer["id"], "policy_config.fast_tokenizer.id"),
                    revision=string(tokenizer["revision"], "policy_config.fast_tokenizer.revision"),
                    allow_patterns=tuple(sorted(expected_files)),
                    local_files_only=True,
                )
            ).resolve()
            _validate_fast_tokenizer(fast_snapshot, expected_files, string(tokenizer["revision"], "policy_config.fast_tokenizer.revision"))
            factory = train_module.ModelTransformFactory(fast_model_tokenizer_kwargs={"fast_tokenizer_path": str(fast_snapshot)})
            train_config = dataclasses.replace(train_config, data=dataclasses.replace(train_config.data, model_transforms=factory))
            sample_kwargs = {"temperature": 0.0, "max_decoding_steps": integer(config.value["max_decoding_steps"], "policy_config.max_decoding_steps")}
        else:
            sample_kwargs = {"num_steps": integer(config.value["num_steps"], "policy_config.num_steps")}
        with _local_downloads(download_module, paligemma):
            self.policy = policy_module.create_trained_policy(
                train_config,
                checkpoint,
                sample_kwargs=sample_kwargs,
            )
        self._validate_runtime(train_config)
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def _validate_runtime(self, train_config: Any) -> None:
        route = self.config.route
        model = train_config.model
        if model.action_horizon != route.action_horizon:
            raise RuntimeError("OpenPI model action horizon differs")
        expected_dim = integer(self.config.value["action_dim"], "policy_config.action_dim")
        if model.action_dim != expected_dim or getattr(model, "dtype", None) != "bfloat16":
            raise RuntimeError("OpenPI model schema differs")
        model_type = getattr(model.model_type, "value", None)
        expected_type = "pi05" if route.name == "openpi_pi05_droid_jointpos" else "pi0_fast"
        if model_type != expected_type:
            raise RuntimeError("OpenPI model type differs")
        if getattr(self.policy, "_is_pytorch_model", None) is not False:
            raise RuntimeError("OpenPI DROID route requires the pinned JAX checkpoint")
        sample_kwargs = getattr(self.policy, "_sample_kwargs", None)
        if route.name == "openpi_pi05_droid_jointpos":
            if sample_kwargs != {"num_steps": 10}:
                raise RuntimeError("OpenPI pi0.5 sampling settings differ")
        elif sample_kwargs != {"temperature": 0.0, "max_decoding_steps": 256}:
            raise RuntimeError("OpenPI pi0-FAST sampling settings differ")

    def reset(self, payload: Any, session_id: str, request_id: str) -> dict[str, Any]:
        obj = fields(payload, {"policy_seed", "task_id"}, path="policy_reset")
        seed = integer(obj["policy_seed"], "policy_reset.policy_seed", minimum=0)
        task_id = string(obj["task_id"], "policy_reset.task_id")
        with self._lock:
            self._sessions[session_id] = _Session(
                seed,
                task_id,
                OpenPiDroidJointPositionAdapter(self.config.route.action_horizon),
            )
        return {"policy_seed": seed, "task_id": task_id, "sample_index": 0}

    def close_session(self, payload: Any, session_id: str, request_id: str) -> dict[str, bool]:
        fields(payload, set(), path="policy_close")
        with self._lock:
            self._sessions.pop(session_id, None)
        return {"closed": True}

    def smoke(self) -> None:
        task, request = synthetic_request(self.config.route.name, session_id=f"{self.config.route.name}-startup-smoke")
        self.reset({"policy_seed": 1000, "task_id": task}, request.session_id, "startup-smoke-reset")
        result = self.act(request.to_mapping(), request.session_id, request.request_id)
        if result["execution_count"] != 1 or result["start_step"] != 0:
            raise RuntimeError("OpenPI DROID startup smoke returned an invalid action")
        self.close_session({}, request.session_id, "startup-smoke-close")

    def act(self, payload: Any, session_id: str, request_id: str) -> dict[str, Any]:
        request = VLARequest.from_mapping(payload)
        if request.session_id != session_id or request.request_id != request_id:
            raise StrictSchemaError("policy_act: payload and envelope identity mismatch")
        with self._lock:
            if session_id not in self._sessions:
                raise StrictSchemaError("policy session must be reset before act")
            session = self._sessions[session_id]
            if session.last_request_id == request_id:
                if session.last_response is None:
                    raise RuntimeError("policy session has an incomplete idempotency record")
                return session.last_response
            if not session.initialized:
                session.adapter.reset(request.observation)
                session.initialized = True
            step = request.observation.step_index
            if session.pending_native is not None:
                if session.pending_step is None or step < session.pending_step:
                    raise StrictSchemaError("policy_act: observation step moved backward")
                if step == session.pending_step + 1:
                    session.adapter.commit(session.pending_native)
                elif not request.refresh or step != session.pending_step:
                    raise StrictSchemaError("policy_act: previous action is not observed as executed")
                session.pending_native = None
                session.pending_step = None
            if request.refresh:
                session.cache = None
                session.cache_index = 0
            if session.cache is None or session.cache_index >= session.cache.shape[0]:
                encoded = session.adapter.encode(request.observation)
                encoded["task"] = request.instruction
                actions = self._infer(encoded, session.policy_seed, session.call_index)
                session.cache = session.adapter.select_native(actions)
                session.cache_index = 0
                session.call_index += 1
            index = session.cache_index
            result = action_chunk(
                session.cache[index : index + 1],
                spec=session.adapter.action_spec,
                execution_count=1,
                request_id=request.request_id,
                session_id=request.session_id,
                start_step=step,
            )
            session.pending_native = session.cache[index].copy()
            session.pending_step = step
            session.cache_index += 1
            response = result.to_mapping()
            session.last_request_id = request_id
            session.last_response = response
            return response

    def _infer(self, payload: Mapping[str, Any], seed: int, call_index: int) -> np.ndarray:
        images = tuple(payload["images"])
        if len(images) != 2:
            raise StrictSchemaError("policy_act.images: expected external and wrist RGB")
        copied = []
        for image in images:
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise StrictSchemaError("policy_act.images: expected uint8 HWC RGB")
            copied.append(np.array(image, dtype=np.uint8, order="C", copy=True))
        state = np.asarray(payload["state"])
        if state.dtype != np.float32 or state.shape != (8,) or not np.isfinite(state).all():
            raise StrictSchemaError("policy_act.state: expected finite float32[8]")
        observation = {
            "observation/exterior_image_1_left": copied[0],
            "observation/wrist_image_left": copied[1],
            "observation/joint_position": state[:7].copy(),
            "observation/gripper_position": state[7:].copy(),
            "prompt": string(payload["task"], "policy_act.task"),
        }
        if self.config.route.name == "openpi_pi05_droid_jointpos":
            noise = _seeded_noise(
                seed,
                call_index,
                self.config.route.action_horizon,
                integer(self.config.value["action_dim"], "policy_config.action_dim"),
            )
            output = self.policy.infer(observation, noise=noise)
        else:
            output = self.policy.infer(observation)
        actions = np.asarray(mapping(output, "policy_output").get("actions"), dtype=np.float32)
        expected = (self.config.route.action_horizon, 8)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise RuntimeError(f"OpenPI DROID returned invalid action shape {actions.shape}")
        return np.ascontiguousarray(actions, dtype=np.float32)
