from __future__ import annotations

import importlib
import hashlib
import os
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from robot_auto_evolve.agent.api import VLARequest
from robot_auto_evolve.benchmarks.contracts import action_chunk
from robot_auto_evolve.benchmarks.droid import MolmoBotDroidAdapter
from robot_auto_evolve.protocol.schema import StrictSchemaError, fields, integer, mapping, number, string

from .config import PolicyServiceConfig
from .smoke import synthetic_request


MODEL_FILE_SHA256 = "db2c62ccdd6773fb4fffad87ed52d299f5f0fc636290133b16150e942f36576d"
MODEL_FILE_SIZE = 19_992_166_548
TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"
TOKENIZER_REVISION = "f50518eb58dfc750271b273fc113bdfc16ec2280"
TOKENIZER_FILES = {
    "config.json": (727, "5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba"),
    "tokenizer.json": (11_422_654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "tokenizer_config.json": (9_377, "a62ff0a2472a0fa1b8eaabcb57c59b58afa42a22831dc141400b6e0cf2b65ce3"),
}
STATE_Q01 = (-0.8200882077217102, -1.0460078716278076, -1.2745805978775024, -2.864607334136963, -1.0115491151809692, 1.2138986587524414, -2.057372808456421, -0.027562683448195457)
STATE_Q99 = (0.7587710618972778, 0.9406100511550903, 0.9344996809959412, -0.9798629283905029, 0.8359407782554626, 3.0869405269622803, 1.9223058223724365, 0.8661524057388306)
ACTION_Q01 = STATE_Q01[:7] + (0.0,)
ACTION_Q99 = STATE_Q99[:7] + (255.0,)


@dataclass(frozen=True)
class _HistoryFrame:
    step: int
    external: np.ndarray
    wrist: np.ndarray


@dataclass
class _Session:
    policy_seed: int
    task_id: str
    adapter: MolmoBotDroidAdapter
    generator: Any
    initialized: bool = False
    history: deque[_HistoryFrame] = field(default_factory=lambda: deque(maxlen=9))
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
        raise RuntimeError(f"MolmoBot source working tree is dirty: {dirty.splitlines()[0]}")


def _float_tuple(value: Any, width: int, path: str) -> tuple[float, ...]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise RuntimeError(f"{path} must contain {width} finite values")
    return tuple(float(item) for item in result)


def _validate_checkpoint(snapshot: Path) -> None:
    required = {".gitattributes", "README.md", "config.yaml", "model.pt"}
    if any(not (snapshot / name).is_file() for name in required):
        raise RuntimeError("MolmoBot checkpoint snapshot is incomplete")
    model_file = (snapshot / "model.pt").resolve()
    if model_file.name != MODEL_FILE_SHA256 or model_file.stat().st_size != MODEL_FILE_SIZE:
        raise RuntimeError("MolmoBot checkpoint weight identity differs")
    value = mapping(yaml.safe_load((snapshot / "config.yaml").read_text(encoding="utf-8")), "checkpoint")
    model = mapping(value.get("model"), "checkpoint.model")
    expected = {
        "model_name": "molmobot",
        "action_dim": 8,
        "action_horizon": 16,
        "n_action_steps": 8,
        "n_obs_steps": 2,
        "obs_step_delta": 8,
        "flow_matching_num_steps": 10,
        "states_mode": "cross_attn",
        "action_expert_layer_mode": "per_layer",
    }
    if {name: model.get(name) for name in expected} != expected:
        raise RuntimeError("MolmoBot checkpoint model contract differs")
    expert = mapping(model.get("action_expert"), "checkpoint.model.action_expert")
    expert_expected = {"max_horizon": 32, "action_dim": 8, "hidden_size": 768, "num_layers": 36, "num_heads": 8}
    if {name: expert.get(name) for name in expert_expected} != expert_expected:
        raise RuntimeError("MolmoBot checkpoint action expert contract differs")
    for processor_name in ("robot_preprocessor", "robot_postprocessor"):
        processor = mapping(model.get(processor_name), f"checkpoint.model.{processor_name}")
        if (
            processor.get("default_repo_id") != "synthmanip"
            or processor.get("action_key") != "action"
            or processor.get("state_keys") != ["observation.state"]
            or processor.get("action_norm_mode") != "quantiles"
            or processor.get("state_norm_mode") != "quantiles"
        ):
            raise RuntimeError(f"MolmoBot checkpoint {processor_name} contract differs")
        stats = mapping(mapping(processor.get("stats_by_repo"), f"checkpoint.model.{processor_name}.stats_by_repo").get("synthmanip"), f"checkpoint.model.{processor_name}.stats_by_repo.synthmanip")
        state_stats = mapping(stats.get("observation.state"), f"checkpoint.model.{processor_name}.observation.state")
        action_stats = mapping(stats.get("action"), f"checkpoint.model.{processor_name}.action")
        if _float_tuple(state_stats.get("q01"), 8, "state.q01") != STATE_Q01 or _float_tuple(state_stats.get("q99"), 8, "state.q99") != STATE_Q99:
            raise RuntimeError(f"MolmoBot checkpoint {processor_name} state statistics differ")
        if _float_tuple(action_stats.get("q01"), 8, "action.q01") != ACTION_Q01 or _float_tuple(action_stats.get("q99"), 8, "action.q99") != ACTION_Q99:
            raise RuntimeError(f"MolmoBot checkpoint {processor_name} action statistics differ")


def _validate_tokenizer(snapshot: Path) -> None:
    if snapshot.name != TOKENIZER_REVISION:
        raise RuntimeError("MolmoBot tokenizer revision differs")
    for name, (size, expected_sha256) in TOKENIZER_FILES.items():
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"MolmoBot tokenizer file differs: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"MolmoBot tokenizer file differs: {name}")


def _strict_wrapper_class(source: Path, tokenizer_snapshot: Path) -> type:
    package_root = source / "MolmoBot"
    expected = package_root / "olmo" / "models" / "molmobot" / "inference_wrapper.py"
    if not expected.is_file():
        raise RuntimeError("MolmoBot inference source is incomplete")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    module = importlib.import_module("olmo.models.molmobot.inference_wrapper")
    if Path(module.__file__).resolve() != expected.resolve():
        raise RuntimeError("MolmoBot imported inference source differs from the pinned checkout")
    base = module.SynthManipMolmoInferenceWrapper

    class StrictSynthManipMolmoInferenceWrapper(base):
        def _load_checkpoint(self) -> None:
            import torch
            from olmo.models.model_config import BaseModelConfig
            from olmo.train.checkpointer import load_model_state
            from olmo.util import resource_path

            config_path = resource_path(self.checkpoint_path, "config.yaml")
            self.model_config = BaseModelConfig.load(config_path, key="model")
            tokenizer = self.model_config.llm.tokenizer
            if tokenizer.identifier != TOKENIZER_ID or tokenizer.tokenizer_dir is not None:
                raise RuntimeError("MolmoBot checkpoint tokenizer contract differs")
            tokenizer.identifier = str(tokenizer_snapshot)
            self.max_seq_len = self.model_config.llm.max_sequence_length
            self.action_horizon = getattr(self.model_config, "action_horizon", 16)
            self.action_dim = getattr(self.model_config, "action_dim", 7)
            self.n_obs_steps = getattr(self.model_config, "n_obs_steps", 1)
            if self.num_flow_steps is None:
                self.num_flow_steps = getattr(self.model_config, "flow_matching_num_steps", 10)
            if self.states_mode is not None:
                self.model_config.states_mode = self.states_mode
            with torch.device("meta"):
                self.model = self.model_config.build_model()
            if self.use_bfloat16:
                self.model.to(torch.bfloat16)
            self.model.to_empty(device=self.device)
            load_model_state(self.checkpoint_path, self.model)
            if self.use_bfloat16:
                self.model.to(self.device, dtype=torch.bfloat16)
            else:
                self.model.to(self.device)
            self.model.eval()
            if self.compile_model:
                self.model.generate_actions = torch.compile(self.model.generate_actions, mode="max-autotune")

        def _normalize_state(self, state: np.ndarray) -> np.ndarray:
            if self.state_preprocessor is None or self.norm_repo_id not in self.state_preprocessor.state_normalizers:
                raise RuntimeError("MolmoBot state normalizer is unavailable")
            result = self.state_preprocessor.normalize_state(state, self.norm_repo_id)
            value = np.asarray(result, dtype=np.float32)
            if value.shape != state.shape or not np.isfinite(value).all():
                raise RuntimeError("MolmoBot state normalization returned invalid values")
            return value

        def _unnormalize_action(self, actions: np.ndarray) -> np.ndarray:
            if self.action_postprocessor is None or self.norm_repo_id not in self.action_postprocessor.action_normalizers:
                raise RuntimeError("MolmoBot action normalizer is unavailable")
            result = self.action_postprocessor.unnormalize_action(actions, self.norm_repo_id)
            value = np.asarray(result, dtype=np.float32)
            if value.shape != actions.shape or not np.isfinite(value).all():
                raise RuntimeError("MolmoBot action unnormalization returned invalid values")
            return value

    return StrictSynthManipMolmoInferenceWrapper


def _validate_wrapper(wrapper: Any) -> None:
    config = wrapper.model_config
    expected = {
        "action_dim": 8,
        "action_horizon": 16,
        "n_action_steps": 8,
        "n_obs_steps": 2,
        "obs_step_delta": 8,
        "flow_matching_num_steps": 10,
        "states_mode": "cross_attn",
        "action_expert_layer_mode": "per_layer",
    }
    if {name: getattr(config, name, None) for name in expected} != expected:
        raise RuntimeError("MolmoBot loaded runtime contract differs")
    tokenizer = config.llm.tokenizer
    if tokenizer.tokenizer_dir is not None:
        raise RuntimeError("MolmoBot loaded tokenizer cache contract differs")
    _validate_tokenizer(Path(tokenizer.identifier).resolve())
    if wrapper.num_flow_steps != 10 or wrapper.norm_repo_id != "synthmanip" or wrapper.use_bfloat16 is not True:
        raise RuntimeError("MolmoBot loaded inference settings differ")


def _history_images(history: deque[_HistoryFrame], current_step: int, delta: int = 8) -> tuple[np.ndarray, ...]:
    by_step = {item.step: item for item in history}
    selected = [by_step[step] for step in (current_step - delta, current_step) if step in by_step]
    if not selected or selected[-1].step != current_step:
        raise RuntimeError("MolmoBot observation history lacks the current frame")
    return tuple([item.external for item in selected] + [item.wrist for item in selected])


def _clamp_joint_action(action: np.ndarray, state: np.ndarray, maximum_delta: float) -> np.ndarray:
    selected = np.asarray(action, dtype=np.float32)
    current = np.asarray(state, dtype=np.float32)
    if selected.shape != (8,) or current.shape != (8,) or not np.isfinite(selected).all() or not np.isfinite(current).all():
        raise RuntimeError("MolmoBot safety clamp requires finite 8D action and state")
    if not np.isfinite(maximum_delta) or maximum_delta <= 0.0:
        raise RuntimeError("MolmoBot safety clamp requires a positive finite delta")
    result = selected.copy()
    delta = result[:7] - current[:7]
    scale = float(np.max(np.abs(delta) / maximum_delta))
    if scale > 1.0:
        result[:7] = current[:7] + delta / scale
    return result


class MolmoBotDroidPolicyBackend:
    def __init__(self, config: PolicyServiceConfig, source_root: str | Path, device: str) -> None:
        if config.route.name != "molmobot_droid" or config.route.backend != "molmobot":
            raise StrictSchemaError("MolmoBot backend received a different route")
        self.config = config
        source = Path(source_root).resolve()
        if _git_head(source) != config.route.source_commit:
            raise RuntimeError("MolmoBot source revision mismatch")
        _require_clean(source)
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("MolmoBot inference requires offline mode")
        torch = importlib.import_module("torch")
        hub = importlib.import_module("huggingface_hub")
        if device != "cuda:0" or torch.cuda.device_count() != 1:
            raise RuntimeError("MolmoBot replica requires exactly one visible GPU as cuda:0")
        checkpoint = mapping(config.value["checkpoint"], "policy_config.checkpoint")
        snapshot = Path(
            hub.snapshot_download(
                repo_id=string(checkpoint["id"], "policy_config.checkpoint.id"),
                revision=string(checkpoint["revision"], "policy_config.checkpoint.revision"),
                local_files_only=True,
            )
        ).resolve()
        if snapshot.name != config.route.revision:
            raise RuntimeError("MolmoBot cached checkpoint revision mismatch")
        _validate_checkpoint(snapshot)
        tokenizer = mapping(config.value["tokenizer"], "policy_config.tokenizer")
        tokenizer_snapshot = Path(
            hub.snapshot_download(
                repo_id=string(tokenizer["id"], "policy_config.tokenizer.id"),
                revision=string(tokenizer["revision"], "policy_config.tokenizer.revision"),
                local_files_only=True,
                allow_patterns=tuple(TOKENIZER_FILES),
            )
        ).resolve()
        _validate_tokenizer(tokenizer_snapshot)
        wrapper_class = _strict_wrapper_class(source, tokenizer_snapshot)
        self.device = torch.device(device)
        self.wrapper = wrapper_class(
            checkpoint_path=str(snapshot),
            device=device,
            num_flow_steps=integer(config.value["num_steps"], "policy_config.num_steps", minimum=1),
            norm_repo_id="synthmanip",
            use_bfloat16=True,
            compile_model=False,
            states_mode=string(config.value["states_mode"], "policy_config.states_mode"),
        )
        _validate_wrapper(self.wrapper)
        self.torch = torch
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def reset(self, payload: Any, session_id: str, request_id: str) -> dict[str, Any]:
        obj = fields(payload, {"policy_seed", "task_id"}, path="policy_reset")
        seed = integer(obj["policy_seed"], "policy_reset.policy_seed", minimum=0)
        task_id = string(obj["task_id"], "policy_reset.task_id")
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        with self._lock:
            self._sessions[session_id] = _Session(seed, task_id, MolmoBotDroidAdapter(), generator)
        return {"policy_seed": seed, "task_id": task_id, "sample_index": 0}

    def close_session(self, payload: Any, session_id: str, request_id: str) -> dict[str, bool]:
        fields(payload, set(), path="policy_close")
        with self._lock:
            self._sessions.pop(session_id, None)
        return {"closed": True}

    def smoke(self) -> None:
        task_id, request = synthetic_request("molmobot_droid", session_id="molmobot-droid-startup-smoke")
        self.reset({"policy_seed": 1000, "task_id": task_id}, request.session_id, "startup-smoke-reset")
        result = self.act(request.to_mapping(), request.session_id, request.request_id)
        if result["execution_count"] != 1 or result["start_step"] != 0:
            raise RuntimeError("MolmoBot startup smoke returned an invalid action")
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
            encoded = session.adapter.encode(request.observation)
            images = tuple(encoded["images"])
            state = np.asarray(encoded["state"], dtype=np.float32)
            if len(images) != 2:
                raise StrictSchemaError("policy_act.images: expected external and wrist RGB")
            frame = _HistoryFrame(step, images[0], images[1])
            if session.history and session.history[-1].step == step:
                session.history[-1] = frame
            else:
                if session.history and step != session.history[-1].step + 1:
                    raise StrictSchemaError("policy_act: observation step is not consecutive")
                session.history.append(frame)
            if request.refresh:
                session.cache = None
                session.cache_index = 0
            execution_count = integer(self.config.value["execution_count"], "policy_config.execution_count", minimum=1)
            if session.cache is None or session.cache_index >= execution_count:
                actions = self._infer(
                    _history_images(session.history, step),
                    state,
                    request.instruction,
                    session.generator,
                )
                session.cache = session.adapter.select_native(actions)
                session.cache_index = 0
            selected = _clamp_joint_action(
                session.cache[session.cache_index],
                state,
                number(self.config.value["max_joint_delta"], "policy_config.max_joint_delta", minimum=0.0),
            )
            result = action_chunk(
                selected[None, :],
                spec=session.adapter.action_spec,
                execution_count=1,
                request_id=request.request_id,
                session_id=request.session_id,
                start_step=step,
            )
            session.pending_native = selected.copy()
            session.pending_step = step
            session.cache_index += 1
            response = result.to_mapping()
            session.last_request_id = request_id
            session.last_response = response
            return response

    def _infer(self, images: tuple[np.ndarray, ...], state: np.ndarray, task: str, generator: Any) -> np.ndarray:
        if len(images) not in {2, 4}:
            raise StrictSchemaError("policy_act.images: expected one or two frames per camera")
        writable_images = []
        for image in images:
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise StrictSchemaError("policy_act.images: expected uint8 HWC RGB")
            writable_images.append(np.array(image, dtype=np.uint8, order="C", copy=True))
        if state.shape != (8,) or state.dtype != np.float32 or not np.isfinite(state).all():
            raise StrictSchemaError("policy_act.state: expected finite float32[8]")
        with self.torch.inference_mode():
            raw = self.wrapper.get_action_chunk(
                images=writable_images,
                task_description=string(task, "policy_act.task"),
                state=state,
                generator=generator,
            )
        actions = np.asarray(raw, dtype=np.float32)
        if actions.shape != (16, 8) or not np.isfinite(actions).all():
            raise RuntimeError(f"MolmoBot returned invalid action shape {actions.shape}")
        return np.ascontiguousarray(actions, dtype=np.float32)
