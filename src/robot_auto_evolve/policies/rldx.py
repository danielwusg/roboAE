from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_auto_evolve.agent.api import VLARequest
from robot_auto_evolve.benchmarks.contracts import action_chunk
from robot_auto_evolve.benchmarks.robocasa365 import (
    ACTION_KEYS,
    CAMERA_KEYS,
    EXECUTION_HORIZON,
    MODEL_HORIZON,
    RLDXRoboCasa365Adapter,
    STATE_KEYS,
    TARGET_TASK_HORIZONS,
    VIDEO_DELTA_INDICES,
)
from robot_auto_evolve.protocol.schema import StrictSchemaError, fields, integer, string

from .config import PolicyServiceConfig
from .xvla import deterministic_seed


MODEL_ID = "RLWRLD/RLDX-1-FT-RC365"
CHECKPOINT_REVISION = "587e9ecdcc5e7184fcc17f58713908edff5af041"
VLM_MODEL_ID = "RLWRLD/RLDX-1-VLM"
VLM_CHECKPOINT_REVISION = "4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7"
RLDX_SOURCE_COMMIT = "ebbfb4f6214bb38de07da1a70f597201feceb6da"
IMAGE_MAX_AREA = 65_536
IMAGE_RESIZE_MULTIPLE = 32
SHARDS = {
    "model-00001-of-00003.safetensors": (
        "2a2f48bd2d2979979700c85c44051c37c3256de528842d82883ba756b070e541",
        4_912_540_968,
    ),
    "model-00002-of-00003.safetensors": (
        "4bb91b9038d7825809c09da425d5dbd6a52ba1a1af25de09bc94ff218e1f80fc",
        4_446_192_352,
    ),
    "model-00003-of-00003.safetensors": (
        "f348bb0aee031e6fd32cad2ff51aa5e4eaf74fa42299dae051f7e3ca0b8adf53",
        4_467_155_576,
    ),
}
VLM_FILES = {
    "added_tokens.json": (59_083, "5d2f525c0a6ecbe1aeb24557bb962ec5f96c21d64816b5e8c29c8055aaf3bf11"),
    "chat_template.jinja": (5_292, "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"),
    "config.json": (1_591, "682bfd9abd200c50dee97346211a437ef0f68f498e427f9e6e5fda3750232056"),
    "generation_config.json": (199, "a0acd9cf47909f86b54951dbf2cfd2af3b0b36178211926760a3a27edb49378d"),
    "merges.txt": (1_671_853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "preprocessor_config.json": (390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
    "special_tokens_map.json": (613, "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
    "tokenizer_config.json": (377_581, "afaa8c12c3ef968a10c298c38195ea5a2b79f1037f2e476554d843af51f391bf"),
    "video_preprocessor_config.json": (385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"),
    "vocab.json": (2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}


@dataclass
class _Session:
    policy_seed: int
    task_id: str
    adapter: RLDXRoboCasa365Adapter
    call_index: int = 0
    initialized: bool = False
    history: list[dict[str, Any]] | None = None
    cache: np.ndarray | None = None
    cache_index: int = 0
    pending_step: int | None = None
    last_request_id: str | None = None
    last_response: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []


def _head(source: Path) -> str:
    return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()


def _require_clean(source: Path, name: str) -> None:
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"{name} source working tree is dirty: {dirty.splitlines()[0]}")


def _validate_checkpoint(snapshot: Path) -> None:
    required = {
        "config.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
        "processor_config.json",
        "statistics.json",
        *SHARDS,
    }
    if any(not (snapshot / name).is_file() for name in required):
        raise RuntimeError("RLDX checkpoint snapshot is incomplete")
    for name, (digest, size) in SHARDS.items():
        resolved = (snapshot / name).resolve()
        if resolved.name != digest or resolved.stat().st_size != size:
            raise RuntimeError(f"RLDX checkpoint shard identity differs: {name}")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if index.get("metadata") != {"total_parameters": 6_912_894_784, "total_size": 13_825_888_896}:
        raise RuntimeError("RLDX checkpoint tensor index metadata differs")
    if set(index.get("weight_map", {}).values()) != set(SHARDS):
        raise RuntimeError("RLDX checkpoint tensor index shard map differs")
    model = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    expected_model = {
        "dtype": "bfloat16",
        "load_bf16": True,
        "model_name": "RLWRLD/RLDX-1-VLM",
        "use_relative_action": True,
        "use_video": True,
        "video_length": 4,
        "memory_video_delta_indices": [-48, -32, -16, 0],
    }
    if {key: model.get(key) for key in expected_model} != expected_model:
        raise RuntimeError("RLDX checkpoint model config differs")
    processor = json.loads((snapshot / "processor_config.json").read_text(encoding="utf-8"))
    kwargs = processor.get("processor_kwargs", {})
    modality = kwargs.get("modality_configs", {}).get("general_embodiment", {})
    if (
        "image_max_area" not in kwargs
        or kwargs["image_max_area"] is not None
        or kwargs.get("image_resize_m") != IMAGE_RESIZE_MULTIPLE
        or kwargs.get("max_action_horizon") != MODEL_HORIZON
        or kwargs.get("max_action_dim") != 64
        or kwargs.get("max_state_dim") != 64
        or kwargs.get("use_relative_action") is not True
        or tuple(modality.get("video", {}).get("delta_indices", ())) != VIDEO_DELTA_INDICES
        or tuple(modality.get("video", {}).get("modality_keys", ())) != CAMERA_KEYS
        or tuple(modality.get("state", {}).get("modality_keys", ())) != STATE_KEYS
        or tuple(modality.get("action", {}).get("modality_keys", ())) != tuple(name for name, _ in ACTION_KEYS)
    ):
        raise RuntimeError("RLDX checkpoint processor config differs")
    statistics = json.loads((snapshot / "statistics.json").read_text(encoding="utf-8"))
    general = statistics.get("general_embodiment", {})
    expected_state_dims = dict(zip(STATE_KEYS, (3, 4, 2, 3, 4), strict=True))
    expected_action_dims = dict(ACTION_KEYS)
    for section, dimensions in (("state", expected_state_dims), ("action", expected_action_dims)):
        actual = general.get(section, {})
        if set(actual) != set(dimensions):
            raise RuntimeError(f"RLDX checkpoint {section} statistics keys differ")
        if any(len(actual[name].get("mean", ())) != width for name, width in dimensions.items()):
            raise RuntimeError(f"RLDX checkpoint {section} statistics dimensions differ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_vlm_snapshot(snapshot: Path) -> None:
    if snapshot.name != VLM_CHECKPOINT_REVISION:
        raise RuntimeError("RLDX VLM runtime revision mismatch")
    for name, (size, digest) in VLM_FILES.items():
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != size or _sha256(path) != digest:
            raise RuntimeError(f"RLDX VLM runtime file differs: {name}")
    if any(snapshot.glob("*.safetensors")):
        raise RuntimeError("RLDX VLM runtime snapshot unexpectedly contains model weights")


def _same_snapshot(requested: Any, expected: Path) -> bool:
    try:
        return Path(os.fspath(requested)).resolve() == expected.resolve()
    except TypeError:
        return False


@contextmanager
def _exact_runtime_loads(transformers: Any, checkpoint: Path, vlm_snapshot: Path):
    config_class = transformers.AutoConfig
    processor_class = transformers.AutoProcessor
    config_descriptor = inspect.getattr_static(config_class, "from_pretrained")
    processor_descriptor = inspect.getattr_static(processor_class, "from_pretrained")
    config_original = config_class.from_pretrained
    processor_original = processor_class.from_pretrained
    counts = {"vlm_config": 0, "vlm_processor": 0, "checkpoint_processor": 0}

    def redirect_config(original: Any):
        def load(cls: Any, name: Any, *args: Any, **kwargs: Any) -> Any:
            if name == VLM_MODEL_ID:
                revision = kwargs.pop("revision", None)
                if revision not in {None, VLM_CHECKPOINT_REVISION}:
                    raise RuntimeError("RLDX VLM runtime requested a different revision")
                name = str(vlm_snapshot)
                kwargs["local_files_only"] = True
                counts["vlm_config"] += 1
            return original(name, *args, **kwargs)

        return classmethod(load)

    def redirect_processor(original: Any):
        def load(cls: Any, name: Any, *args: Any, **kwargs: Any) -> Any:
            if _same_snapshot(name, checkpoint):
                requested_area = kwargs.get("image_max_area")
                requested_multiple = kwargs.get("image_resize_m")
                if requested_area not in {None, IMAGE_MAX_AREA}:
                    raise RuntimeError("RLDX processor requested a different image area")
                if requested_multiple not in {None, IMAGE_RESIZE_MULTIPLE}:
                    raise RuntimeError("RLDX processor requested a different resize multiple")
                kwargs["image_max_area"] = IMAGE_MAX_AREA
                kwargs["image_resize_m"] = IMAGE_RESIZE_MULTIPLE
                counts["checkpoint_processor"] += 1
            elif name == VLM_MODEL_ID:
                revision = kwargs.pop("revision", None)
                if revision not in {None, VLM_CHECKPOINT_REVISION}:
                    raise RuntimeError("RLDX VLM runtime requested a different revision")
                name = str(vlm_snapshot)
                kwargs["local_files_only"] = True
                counts["vlm_processor"] += 1
            return original(name, *args, **kwargs)

        return classmethod(load)

    setattr(config_class, "from_pretrained", redirect_config(config_original))
    setattr(processor_class, "from_pretrained", redirect_processor(processor_original))
    try:
        yield counts
    finally:
        setattr(config_class, "from_pretrained", config_descriptor)
        setattr(processor_class, "from_pretrained", processor_descriptor)


class RLDXRoboCasa365PolicyBackend:
    def __init__(self, config: PolicyServiceConfig, source_root: str | Path, device: str) -> None:
        if not isinstance(config, PolicyServiceConfig) or config.route.name != "rldx_robocasa365":
            raise StrictSchemaError("RLDX backend requires the RLDX RoboCasa365 policy config")
        source = Path(source_root).resolve()
        if not (source / "rldx" / "policy" / "rldx_policy.py").is_file() or _head(source) != RLDX_SOURCE_COMMIT:
            raise RuntimeError("RLDX source checkout mismatch")
        _require_clean(source, "RLDX")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        if os.environ.get("NO_ALBUMENTATIONS_UPDATE") not in {None, "1"}:
            raise RuntimeError("NO_ALBUMENTATIONS_UPDATE must be 1 for offline RLDX inference")
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
        torch = importlib.import_module("torch")
        hub = importlib.import_module("huggingface_hub")
        rldx = importlib.import_module("rldx")
        transformers = importlib.import_module("transformers")
        try:
            Path(rldx.__file__).resolve().relative_to(source)
        except ValueError as exc:
            raise RuntimeError("imported RLDX package differs from the pinned checkout") from exc
        if torch.__version__.split("+", 1)[0] != "2.7.0" or transformers.__version__ != "4.57.0":
            raise RuntimeError("RLDX policy dependency version mismatch")
        if device != "cuda:0" or torch.cuda.device_count() != 1:
            raise RuntimeError("RLDX replica requires exactly one visible GPU as cuda:0")
        snapshot = Path(
            hub.snapshot_download(repo_id=MODEL_ID, revision=CHECKPOINT_REVISION, local_files_only=True)
        ).resolve()
        if snapshot.name != CHECKPOINT_REVISION:
            raise RuntimeError("RLDX cached checkpoint revision mismatch")
        _validate_checkpoint(snapshot)
        vlm_snapshot = Path(
            hub.snapshot_download(
                repo_id=VLM_MODEL_ID,
                revision=VLM_CHECKPOINT_REVISION,
                local_files_only=True,
            )
        ).resolve()
        _validate_vlm_snapshot(vlm_snapshot)
        tags = importlib.import_module("rldx.data.embodiment_tags")
        policy_module = importlib.import_module("rldx.policy.rldx_policy")
        self.device = torch.device(device)
        with _exact_runtime_loads(transformers, snapshot, vlm_snapshot) as exact_loads:
            self.policy = policy_module.RLDXPolicy(
                model_path=str(snapshot),
                embodiment_tag=tags.EmbodimentTag.GENERAL_EMBODIMENT,
                device=device,
                strict=True,
            )
        if any(exact_loads[name] < 1 for name in exact_loads):
            raise RuntimeError("RLDX exact runtime load was not exercised")
        processor = self.policy.processor
        if (
            processor.image_max_area != IMAGE_MAX_AREA
            or processor.image_resize_m != IMAGE_RESIZE_MULTIPLE
            or processor.train_image_transform.transforms[0].max_area != IMAGE_MAX_AREA
            or processor.eval_image_transform.transforms[0].max_area != IMAGE_MAX_AREA
        ):
            raise RuntimeError("RLDX image processor compatibility override failed")
        self.wrapper = policy_module.RLDXSimPolicyWrapper(self.policy, strict=True)
        self.config = config
        self.torch = torch
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def reset(self, payload: Any, session_id: str, request_id: str) -> dict[str, Any]:
        obj = fields(payload, {"policy_seed", "task_id"}, path="policy_reset")
        seed = integer(obj["policy_seed"], "policy_reset.policy_seed", minimum=0)
        task_id = string(obj["task_id"], "policy_reset.task_id")
        if task_id not in TARGET_TASK_HORIZONS:
            raise StrictSchemaError("policy_reset.task_id: unsupported for RLDX RoboCasa365")
        with self._lock:
            self.wrapper.reset({"session_ids": [session_id]})
            self._sessions[session_id] = _Session(seed, task_id, RLDXRoboCasa365Adapter())
        return {"policy_seed": seed, "task_id": task_id, "sample_index": 0}

    def close_session(self, payload: Any, session_id: str, request_id: str) -> dict[str, bool]:
        fields(payload, set(), path="policy_close")
        with self._lock:
            self.wrapper.reset({"session_ids": [session_id]})
            self._sessions.pop(session_id, None)
        return {"closed": True}

    def smoke(self) -> None:
        session_id = "rldx-startup-smoke"
        payload: dict[str, Any] = {
            f"video.{key}": np.zeros((1, 4, 256, 256, 3), dtype=np.uint8) for key in CAMERA_KEYS
        }
        for key, width in zip(STATE_KEYS, (3, 4, 2, 3, 4), strict=True):
            payload[f"state.{key}"] = np.zeros((1, 1, width), dtype=np.float32)
        payload["annotation.human.task_description"] = ["open the cabinet"]
        result = self._infer(payload, deterministic_seed(0, 0), session_id, True)
        selected = RLDXRoboCasa365Adapter().select_native(result)
        if selected.shape != (EXECUTION_HORIZON, 12):
            raise RuntimeError("RLDX startup smoke returned an invalid action")
        self.wrapper.reset({"session_ids": [session_id]})

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
            if session.pending_step is not None:
                if request.refresh:
                    if step not in {session.pending_step, session.pending_step + 1}:
                        raise StrictSchemaError("policy_act: refresh skipped an unobserved action step")
                elif step != session.pending_step + 1:
                    raise StrictSchemaError("policy_act: previous action is not observed as executed")
                session.pending_step = None
            encoded = session.adapter.encode_current(request.observation)
            encoded["language"] = request.instruction
            history = session.history
            if history is None:
                raise RuntimeError("RLDX session history is absent")
            if history and step == history[-1]["step_index"]:
                if not request.refresh:
                    raise StrictSchemaError("policy_act: repeated step requires refresh")
                history[-1] = encoded
            elif history and step == history[-1]["step_index"] + 1:
                history.append(encoded)
            elif not history and step == 0:
                history.append(encoded)
            else:
                raise StrictSchemaError("policy_act: RLDX observations must be contiguous from step zero")
            if len(history) > 7:
                del history[:-7]
            if request.refresh:
                session.cache = None
                session.cache_index = 0
            if session.cache is None or session.cache_index >= session.cache.shape[0]:
                temporal = session.adapter.temporal_batch(history)
                native = self._infer(
                    temporal,
                    deterministic_seed(session.policy_seed, session.call_index),
                    session_id,
                    session.call_index == 0,
                )
                session.cache = session.adapter.select_native(native)
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
            session.cache_index += 1
            session.pending_step = step
            response = result.to_mapping()
            session.last_request_id = request_id
            session.last_response = response
            return response

    def _infer(
        self,
        payload: Mapping[str, Any],
        seed: int,
        session_id: str,
        reset_memory: bool,
    ) -> Mapping[str, Any]:
        torch = self.torch
        with torch.random.fork_rng(devices=[self.device.index]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            with torch.inference_mode():
                action, _ = self.wrapper.get_action(
                    dict(payload),
                    options={"reset_memory": [reset_memory], "session_ids": [session_id]},
                )
        return action
