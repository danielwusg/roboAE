from __future__ import annotations

import importlib
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_auto_evolve.agent.api import VLARequest
from robot_auto_evolve.benchmarks.contracts import action_chunk
from robot_auto_evolve.benchmarks.pi05 import EXECUTION_HORIZON, MODEL_HORIZON, Pi05LiberoAdapter
from robot_auto_evolve.benchmarks.libero_pro import libero_bare_task
from robot_auto_evolve.benchmarks.xvla import LIBERO_TASKS
from robot_auto_evolve.protocol.schema import StrictSchemaError, fields, integer, mapping, string

from .config import PolicyServiceConfig
from .smoke import synthetic_request
from .xvla import deterministic_seed


TRANSFORMERS_COMMIT = "dcddb970176382c0fcf4521b0c0e6fc15894dfe0"
TOKENIZER_ID = "google/paligemma-3b-pt-224"
TOKENIZER_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"
TOKENIZER_FILES = (
    "added_tokens.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
CHECKPOINT_ALIASES = {
    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight":
        "model.paligemma_with_expert.paligemma.lm_head.weight",
}


@dataclass
class _Session:
    policy_seed: int
    task_id: str
    adapter: Pi05LiberoAdapter
    call_index: int = 0
    initialized: bool = False
    cache: np.ndarray | None = None
    cache_index: int = 0
    pending_step: int | None = None
    last_request_id: str | None = None
    last_response: dict[str, Any] | None = None


def _head(source: Path) -> str:
    return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()


def _require_clean(source: Path, name: str) -> None:
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"{name} source working tree is dirty: {dirty.splitlines()[0]}")


def _inside(module: Any, source: Path) -> bool:
    try:
        Path(module.__file__).resolve().relative_to(source.resolve())
    except (AttributeError, ValueError):
        return False
    return True


def _validate_checkpoint_config(config: Any) -> None:
    input_shapes = {key: tuple(value.shape) for key, value in config.input_features.items()}
    expected_inputs = {
        "observation.images.empty_camera_0": (3, 224, 224),
        "observation.images.image": (3, 256, 256),
        "observation.images.image2": (3, 256, 256),
        "observation.state": (8,),
    }
    output_shapes = {key: tuple(value.shape) for key, value in config.output_features.items()}
    if input_shapes != expected_inputs or output_shapes != {"action": (7,)}:
        raise RuntimeError("pi0.5 checkpoint feature schema mismatch")
    expected = {
        "chunk_size": MODEL_HORIZON,
        "n_action_steps": MODEL_HORIZON,
        "num_inference_steps": 10,
        "dtype": "bfloat16",
        "image_resolution": (224, 224),
        "compile_model": True,
        "compile_mode": "max-autotune",
    }
    actual = {name: getattr(config, name) for name in expected}
    actual["image_resolution"] = tuple(actual["image_resolution"])
    if actual != expected:
        raise RuntimeError(f"pi0.5 checkpoint runtime schema mismatch: {actual}")


def _restore_checkpoint_aliases(
    model: Any,
    state: Mapping[str, Any],
    metadata: Mapping[str, str] | None,
) -> dict[str, Any]:
    if metadata != CHECKPOINT_ALIASES:
        raise RuntimeError("pi0.5 checkpoint alias metadata mismatch")
    restored = dict(state)
    for target, source in CHECKPOINT_ALIASES.items():
        if target in restored or source not in restored:
            raise RuntimeError("pi0.5 checkpoint alias tensor mismatch")
        target_parameter = model.get_parameter(target)
        source_parameter = model.get_parameter(source)
        source_tensor = restored[source]
        if (
            target_parameter is not source_parameter
            or tuple(source_tensor.shape) != tuple(source_parameter.shape)
            or source_tensor.dtype != source_parameter.dtype
        ):
            raise RuntimeError("pi0.5 checkpoint alias model mismatch")
        restored[target] = restored[source]
    return restored


def _strict_model(policy_class: Any, config: Any, snapshot: Path, device: Any) -> Any:
    safetensors_package = importlib.import_module("safetensors")
    safetensors = importlib.import_module("safetensors.torch")
    model = policy_class(config)
    weights = snapshot / "model.safetensors"
    with safetensors_package.safe_open(weights, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    state = _restore_checkpoint_aliases(model, safetensors.load_file(weights), metadata)
    fixed = model._fix_pytorch_state_dict_keys(state, model.config)
    remapped = {key if key.startswith("model.") else f"model.{key}": value for key, value in fixed.items()}
    model.load_state_dict(remapped, strict=True)
    del state, fixed, remapped
    return model.to(device).eval()


class Pi05LiberoPolicyBackend:
    def __init__(self, config: PolicyServiceConfig, source_root: str | Path, device: str) -> None:
        if config.route.name != "pi05_libero":
            raise StrictSchemaError("pi0.5 backend received a different route")
        self.config = config
        source = Path(source_root).resolve()
        transformers_source = source.parent / "transformers-pi"
        if not (source / "src" / "lerobot" / "policies" / "pi05").is_dir() or _head(source) != config.route.source_commit:
            raise RuntimeError("LeRobot source checkout mismatch")
        if not (transformers_source / "src" / "transformers" / "models" / "gemma").is_dir() or _head(transformers_source) != TRANSFORMERS_COMMIT:
            raise RuntimeError("pi0.5 Transformers source checkout mismatch")
        _require_clean(source, "LeRobot")
        _require_clean(transformers_source, "pi0.5 Transformers")
        torch = importlib.import_module("torch")
        hub = importlib.import_module("huggingface_hub")
        lerobot = importlib.import_module("lerobot")
        transformers = importlib.import_module("transformers")
        if not _inside(lerobot, source) or not _inside(transformers, transformers_source):
            raise RuntimeError("pi0.5 imported source differs from pinned checkouts")
        if transformers.__version__ != "4.53.3":
            raise RuntimeError("pi0.5 Transformers version mismatch")
        if device != "cuda:0" or torch.cuda.device_count() != 1:
            raise RuntimeError("pi0.5 replica requires exactly one visible GPU as cuda:0")
        checkpoint = mapping(config.value["checkpoint"], "policy_config.checkpoint")
        model_id = string(checkpoint["id"], "policy_config.checkpoint.id")
        revision = string(checkpoint["revision"], "policy_config.checkpoint.revision")
        snapshot = Path(hub.snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)).resolve()
        tokenizer = Path(
            hub.snapshot_download(
                repo_id=TOKENIZER_ID,
                revision=TOKENIZER_REVISION,
                allow_patterns=list(TOKENIZER_FILES),
                local_files_only=True,
            )
        ).resolve()
        if snapshot.name != revision or tokenizer.name != TOKENIZER_REVISION:
            raise RuntimeError("pi0.5 cached artifact revision mismatch")
        if any(not (tokenizer / name).is_file() for name in TOKENIZER_FILES):
            raise RuntimeError("pi0.5 tokenizer snapshot is incomplete")
        policy_configs = importlib.import_module("lerobot.configs.policies")
        modeling = importlib.import_module("lerobot.policies.pi05.modeling_pi05")
        importlib.import_module("lerobot.policies.pi05.processor_pi05")
        pipeline = importlib.import_module("lerobot.processor.pipeline")
        converters = importlib.import_module("lerobot.processor.converters")
        policy_utils = importlib.import_module("lerobot.policies.utils")
        self.device = torch.device(device)
        policy_config = policy_configs.PreTrainedConfig.from_pretrained(snapshot)
        _validate_checkpoint_config(policy_config)
        policy_config.device = str(self.device)
        policy_config.n_action_steps = EXECUTION_HORIZON
        policy_config.compile_mode = string(config.value["compile_mode"], "policy_config.compile_mode")
        self.model = _strict_model(modeling.PI05Policy, policy_config, snapshot, self.device)
        self.preprocess = pipeline.PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=snapshot,
            config_filename="policy_preprocessor.json",
            overrides={
                "device_processor": {"device": str(self.device)},
                "tokenizer_processor": {"tokenizer_name": str(tokenizer)},
            },
            to_transition=converters.batch_to_transition,
            to_output=converters.transition_to_batch,
        )
        self.postprocess = pipeline.PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=snapshot,
            config_filename="policy_postprocessor.json",
            overrides={},
            to_transition=converters.policy_action_to_transition,
            to_output=converters.transition_to_policy_action,
        )
        self.prepare = policy_utils.prepare_observation_for_inference
        self.torch = torch
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def reset(self, payload: Any, session_id: str, request_id: str) -> dict[str, Any]:
        obj = fields(payload, {"policy_seed", "task_id"}, path="policy_reset")
        seed = integer(obj["policy_seed"], "policy_reset.policy_seed", minimum=0)
        task_id = string(obj["task_id"], "policy_reset.task_id")
        if libero_bare_task(task_id) not in LIBERO_TASKS:
            raise StrictSchemaError("policy_reset.task_id: unsupported for pi0.5 LIBERO")
        with self._lock:
            self._sessions[session_id] = _Session(seed, task_id, Pi05LiberoAdapter())
        return {"policy_seed": seed, "task_id": task_id, "sample_index": 0}

    def close_session(self, payload: Any, session_id: str, request_id: str) -> dict[str, bool]:
        fields(payload, set(), path="policy_close")
        with self._lock:
            self._sessions.pop(session_id, None)
        return {"closed": True}

    def smoke(self) -> None:
        task_id, request = synthetic_request(self.config.route.name)
        self.reset({"policy_seed": 0, "task_id": task_id}, request.session_id, "startup-smoke-reset")
        result = self.act(request.to_mapping(), request.session_id, request.request_id)
        if result["execution_count"] != 1 or result["start_step"] != 0:
            raise RuntimeError("pi0.5 startup smoke returned an invalid action")
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
            if session.pending_step is not None:
                if request.refresh:
                    if step not in {session.pending_step, session.pending_step + 1}:
                        raise StrictSchemaError("policy_act: refresh skipped an unobserved action step")
                elif step != session.pending_step + 1:
                    raise StrictSchemaError("policy_act: previous action is not observed as executed")
                session.pending_step = None
            if request.refresh:
                session.cache = None
                session.cache_index = 0
            if session.cache is None or session.cache_index >= session.cache.shape[0]:
                encoded = session.adapter.encode(request.observation)
                encoded["task"] = request.instruction
                native = self._infer(encoded, deterministic_seed(session.policy_seed, session.call_index))
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

    def _infer(self, payload: Mapping[str, Any], seed: int) -> np.ndarray:
        frame = {
            "observation.images.image": np.asarray(payload["observation.images.image"], dtype=np.uint8),
            "observation.images.image2": np.asarray(payload["observation.images.image2"], dtype=np.uint8),
            "observation.state": np.asarray(payload["observation.state"], dtype=np.float32),
        }
        frame = self.prepare(frame, self.device, task=string(payload["task"], "policy_act.task"), robot_type="")
        frame = self.preprocess(frame)
        torch = self.torch
        with torch.random.fork_rng(devices=[self.device.index]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            with torch.inference_mode():
                actions = self.model.predict_action_chunk(frame)
        actions = self.postprocess(actions)
        result = actions.squeeze(0).detach().to("cpu").numpy().astype(np.float32)
        if result.shape != (MODEL_HORIZON, 7) or not np.isfinite(result).all():
            raise RuntimeError(f"pi0.5 returned invalid action shape {result.shape}")
        return np.ascontiguousarray(result)
