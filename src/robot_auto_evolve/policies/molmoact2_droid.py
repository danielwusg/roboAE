from __future__ import annotations

import importlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_auto_evolve.agent.api import VLARequest
from robot_auto_evolve.benchmarks.contracts import action_chunk
from robot_auto_evolve.benchmarks.droid import MolmoAct2DroidAdapter
from robot_auto_evolve.protocol.schema import StrictSchemaError, boolean, fields, integer, mapping, string

from .config import PolicyServiceConfig
from .molmoact2 import _git_head, _require_clean, _validate_remote_code, _validate_snapshot
from .smoke import synthetic_request


@dataclass
class _DroidSession:
    policy_seed: int
    task_id: str
    adapter: MolmoAct2DroidAdapter
    generator: Any
    initialized: bool = False
    inference_count: int = 0
    cache: np.ndarray | None = None
    cache_index: int = 0
    pending_native: np.ndarray | None = None
    pending_step: int | None = None
    last_request_id: str | None = None
    last_response: dict[str, Any] | None = None


def _finite_width(value: Any, width: int, path: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise RuntimeError(f"{path} must contain {width} finite values")
    return result


def _validate_droid_norm_stats(snapshot: Path) -> None:
    value = json.loads((snapshot / "norm_stats.json").read_text(encoding="utf-8"))
    if value.get("format") != "molmoact2_norm_stats.v1" or value.get("norm_mode") != "q01_q99":
        raise RuntimeError("MolmoAct2 DROID normalization format differs")
    metadata = mapping(value.get("metadata_by_tag"), "norm_stats.metadata_by_tag")
    if set(metadata) != {"franka_droid"}:
        raise RuntimeError("MolmoAct2 DROID normalization tags differ")
    droid = mapping(metadata["franka_droid"], "norm_stats.franka_droid")
    expected = {
        "action_key": "action",
        "state_key": "observation.state",
        "camera_keys": [
            "observation.images.exterior_1_left",
            "observation.images.exterior_2_left",
            "observation.images.wrist_left",
        ],
        "normalize_gripper": False,
        "action_horizon": 15,
        "n_action_steps": 15,
        "setup_type": "single franka robotic arm in droid",
        "control_mode": "absolute joint pose",
    }
    for name, expected_value in expected.items():
        if droid.get(name) != expected_value:
            raise RuntimeError(f"MolmoAct2 DROID normalization field {name!r} differs")
    names = [f"joint_{index}" for index in range(7)] + ["gripper"]
    for key in ("action_stats", "state_stats"):
        stats = mapping(droid.get(key), f"norm_stats.franka_droid.{key}")
        if stats.get("names") != names or stats.get("mask") != [True] * 7 + [False]:
            raise RuntimeError(f"MolmoAct2 DROID {key} schema differs")
        minimum = _finite_width(stats.get("min"), 8, f"norm_stats.{key}.min")
        maximum = _finite_width(stats.get("max"), 8, f"norm_stats.{key}.max")
        if minimum[7] != 0.0 or maximum[7] != 1.0 or np.any(minimum >= maximum):
            raise RuntimeError(f"MolmoAct2 DROID {key} ranges differ")


class MolmoAct2DroidPolicyBackend:
    def __init__(self, config: PolicyServiceConfig, source_root: str | Path, device: str) -> None:
        if config.route.name != "molmoact2_droid" or config.route.backend != "molmoact2_droid":
            raise StrictSchemaError("MolmoAct2 DROID backend received a different route")
        self.config = config
        source = Path(source_root).resolve()
        if not (source / "examples" / "droid" / "host_server_droid.py").is_file():
            raise RuntimeError("MolmoAct2 DROID inference source is incomplete")
        if _git_head(source) != config.route.source_commit:
            raise RuntimeError("MolmoAct2 DROID inference source revision mismatch")
        _require_clean(source)
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        hub = importlib.import_module("huggingface_hub")
        if transformers.__version__ != "5.3.0":
            raise RuntimeError("MolmoAct2 DROID requires Transformers 5.3.0")
        if device != "cuda:0" or torch.cuda.device_count() != 1:
            raise RuntimeError("MolmoAct2 DROID replica requires exactly one visible GPU as cuda:0")
        checkpoint = mapping(config.value["checkpoint"], "policy_config.checkpoint")
        model_id = string(checkpoint["id"], "policy_config.checkpoint.id")
        revision = string(checkpoint["revision"], "policy_config.checkpoint.revision")
        snapshot = Path(
            hub.snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_files_only=True,
            )
        ).resolve()
        if snapshot.name != revision:
            raise RuntimeError("MolmoAct2 DROID cached artifact revision mismatch")
        _validate_snapshot(snapshot)
        _validate_droid_norm_stats(snapshot)
        self.device = torch.device(device)
        self.processor = transformers.AutoProcessor.from_pretrained(
            snapshot,
            trust_remote_code=True,
            use_fast=False,
            extra_special_tokens={},
            local_files_only=True,
        )
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            snapshot,
            trust_remote_code=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self.device).eval()
        _validate_remote_code(self.processor, snapshot, "processing_molmoact2.py")
        _validate_remote_code(self.model, snapshot, "modeling_molmoact2.py")
        self._validate_model_config()
        self.torch = torch
        self._sessions: dict[str, _DroidSession] = {}
        self._lock = threading.Lock()

    def _validate_model_config(self) -> None:
        expected = {
            "action_mode": "both",
            "add_action_expert": True,
            "action_expert_depth_gate": False,
            "action_expert_depth_gate_per_layer": False,
            "enable_depth_reasoning": False,
            "flow_matching_num_steps": 10,
            "max_action_dim": 32,
            "max_action_horizon": 15,
            "n_obs_steps": 1,
            "num_depth_tokens": 0,
            "state_format": "discrete",
        }
        actual = {name: getattr(self.model.config, name) for name in expected}
        if actual != expected:
            raise RuntimeError(f"MolmoAct2 DROID checkpoint runtime schema mismatch: {actual}")

    def reset(self, payload: Any, session_id: str, request_id: str) -> dict[str, Any]:
        obj = fields(payload, {"policy_seed", "task_id"}, path="policy_reset")
        seed = integer(obj["policy_seed"], "policy_reset.policy_seed", minimum=0)
        task_id = string(obj["task_id"], "policy_reset.task_id")
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        with self._lock:
            self._sessions[session_id] = _DroidSession(seed, task_id, MolmoAct2DroidAdapter(), generator)
        return {"policy_seed": seed, "task_id": task_id, "sample_index": 0}

    def close_session(self, payload: Any, session_id: str, request_id: str) -> dict[str, bool]:
        fields(payload, set(), path="policy_close")
        with self._lock:
            self._sessions.pop(session_id, None)
        return {"closed": True}

    def smoke(self) -> None:
        task_id, request = synthetic_request("molmoact2_droid", session_id="molmoact2-droid-startup-smoke")
        self.reset({"policy_seed": 1000, "task_id": task_id}, request.session_id, "startup-smoke-reset")
        result = self.act(request.to_mapping(), request.session_id, request.request_id)
        if result["execution_count"] != 1 or result["start_step"] != 0:
            raise RuntimeError("MolmoAct2 DROID startup smoke returned an invalid action")
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
                reason = "refresh" if request.refresh else "cache_exhausted"
                encoded = session.adapter.encode(request.observation)
                encoded["task"] = request.instruction
                actions = self._infer(encoded, session.generator)
                session.cache = session.adapter.select_native(actions)
                session.cache_index = 0
                session.inference_count += 1
                print(
                    json.dumps(
                        {
                            "event": "molmoact2_droid_inference",
                            "inference_index": session.inference_count,
                            "reason": reason,
                            "session_id": session_id,
                            "step_index": step,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
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

    def _infer(self, payload: Mapping[str, Any], generator: Any) -> np.ndarray:
        images = tuple(payload["images"])
        if len(images) != 3:
            raise StrictSchemaError("policy_act.images: expected external, duplicated external, and wrist RGB")
        for image in images:
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise StrictSchemaError("policy_act.images: expected uint8 HWC RGB")
        state_value = np.asarray(payload["state"])
        if state_value.dtype != np.float32 or state_value.shape != (8,) or not np.isfinite(state_value).all():
            raise StrictSchemaError("policy_act.state: expected finite float32[8]")
        config = self.config.value
        with self.torch.inference_mode():
            output = self.model.predict_action(
                processor=self.processor,
                images=list(images),
                task=string(payload["task"], "policy_act.task"),
                state=state_value,
                norm_tag=string(config["norm_tag"], "policy_config.norm_tag"),
                inference_action_mode=string(config["inference_action_mode"], "policy_config.inference_action_mode"),
                enable_depth_reasoning=False,
                num_steps=integer(config["num_steps"], "policy_config.num_steps", minimum=1),
                n_action_steps=integer(config["execution_count"], "policy_config.execution_count", minimum=1),
                generator=generator,
                normalize_language=boolean(config["normalize_language"], "policy_config.normalize_language"),
                enable_cuda_graph=boolean(config["enable_cuda_graph"], "policy_config.enable_cuda_graph"),
            )
        raw = output.actions
        if self.torch.is_tensor(raw):
            raw = raw.detach().to("cpu", dtype=self.torch.float32).numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.shape != (15, 8) or not np.isfinite(actions).all():
            raise RuntimeError(f"MolmoAct2 DROID returned invalid action shape {actions.shape}")
        return np.ascontiguousarray(actions, dtype=np.float32)
