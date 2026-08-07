from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_auto_evolve.agent.api import VLARequest
from robot_auto_evolve.benchmarks.contracts import action_chunk
from robot_auto_evolve.benchmarks.libero_pro import libero_bare_task
from robot_auto_evolve.benchmarks.molmoact2 import MolmoAct2LiberoAdapter
from robot_auto_evolve.benchmarks.xvla import LIBERO_TASKS
from robot_auto_evolve.protocol.schema import StrictSchemaError, boolean, fields, integer, mapping, string

from .config import PolicyServiceConfig
from .smoke import synthetic_request


@dataclass
class _Session:
    policy_seed: int
    task_id: str
    adapter: MolmoAct2LiberoAdapter
    generator: Any
    initialized: bool = False
    cache: np.ndarray | None = None
    cache_index: int = 0
    depth_cache: Mapping[str, Any] | None = None
    pending_native: np.ndarray | None = None
    pending_step: int | None = None
    last_request_id: str | None = None
    last_response: dict[str, Any] | None = None


def _git_head(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def _require_clean(source: Path) -> None:
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"MolmoAct2 inference source is dirty: {dirty.splitlines()[0]}")


def _validate_snapshot(snapshot: Path) -> None:
    required = {
        "config.json",
        "configuration_molmoact2.py",
        "image_processing_molmoact2.py",
        "inference.py",
        "model.safetensors.index.json",
        "modeling_molmoact2.py",
        "norm_stats.json",
        "processing_molmoact2.py",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_processing_molmoact2.py",
    }
    missing = sorted(name for name in required if not (snapshot / name).is_file())
    if missing:
        raise RuntimeError(f"MolmoAct2 checkpoint is incomplete: {missing}")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = {string(name, "checkpoint.weight_map shard") for name in mapping(index["weight_map"], "checkpoint.weight_map").values()}
    if not shards or any(not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0 for name in shards):
        raise RuntimeError("MolmoAct2 checkpoint weight shards are incomplete")


def _validate_remote_code(instance: Any, snapshot: Path, filename: str) -> None:
    loaded = Path(inspect.getfile(instance.__class__)).resolve()
    expected = (snapshot / filename).resolve()
    if loaded.read_bytes() != expected.read_bytes():
        raise RuntimeError(f"MolmoAct2 loaded remote code differs from {filename}")


class MolmoAct2LiberoPolicyBackend:
    def __init__(self, config: PolicyServiceConfig, source_root: str | Path, device: str) -> None:
        if config.route.backend != "molmoact2":
            raise StrictSchemaError("MolmoAct2 backend received a different route")
        self.config = config
        source = Path(source_root).resolve()
        policy_source = source / "src" / "lerobot" / "policies" / "molmoact2"
        if not (policy_source / "modeling_molmoact2.py").is_file():
            raise RuntimeError("MolmoAct2 LeRobot inference source is incomplete")
        if _git_head(source) != config.route.source_commit:
            raise RuntimeError("MolmoAct2 LeRobot inference source revision mismatch")
        _require_clean(source)
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        hub = importlib.import_module("huggingface_hub")
        if transformers.__version__ != "5.3.0":
            raise RuntimeError("MolmoAct2 requires Transformers 5.3.0")
        if device != "cuda:0" or torch.cuda.device_count() != 1:
            raise RuntimeError("MolmoAct2 replica requires exactly one visible GPU as cuda:0")
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
            raise RuntimeError("MolmoAct2 cached artifact revision mismatch")
        _validate_snapshot(snapshot)
        self.device = torch.device(device)
        self.processor = transformers.AutoProcessor.from_pretrained(
            snapshot,
            trust_remote_code=True,
            use_fast=False,
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
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def _validate_model_config(self) -> None:
        expected_depth = self.config.route.name == "molmoact2_think_libero"
        expected = {
            "action_mode": "both",
            "add_action_expert": True,
            "action_expert_depth_gate": expected_depth,
            "action_expert_depth_gate_per_layer": expected_depth,
            "enable_depth_reasoning": expected_depth,
            "flow_matching_num_steps": 10,
            "max_action_dim": 32,
            "max_action_horizon": 10,
            "n_obs_steps": 1,
            "num_depth_tokens": 128 if expected_depth else 0,
            "state_format": "discrete",
        }
        actual = {name: getattr(self.model.config, name) for name in expected}
        if actual != expected:
            raise RuntimeError(f"MolmoAct2 checkpoint runtime schema mismatch: {actual}")
        depth = boolean(
            self.config.value["enable_depth_reasoning"],
            "policy_config.enable_depth_reasoning",
        )
        adaptive = boolean(
            self.config.value["enable_adaptive_depth"],
            "policy_config.enable_adaptive_depth",
        )
        if depth != expected_depth or adaptive != expected_depth:
            raise RuntimeError("MolmoAct2 service depth settings differ from checkpoint")

    def reset(self, payload: Any, session_id: str, request_id: str) -> dict[str, Any]:
        obj = fields(payload, {"policy_seed", "task_id"}, path="policy_reset")
        seed = integer(obj["policy_seed"], "policy_reset.policy_seed", minimum=0)
        task_id = string(obj["task_id"], "policy_reset.task_id")
        if libero_bare_task(task_id) not in LIBERO_TASKS:
            raise StrictSchemaError("policy_reset.task_id: unsupported for MolmoAct2 LIBERO")
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        with self._lock:
            self._sessions[session_id] = _Session(
                seed,
                task_id,
                MolmoAct2LiberoAdapter(),
                generator,
            )
        return {"policy_seed": seed, "task_id": task_id, "sample_index": 0}

    def close_session(self, payload: Any, session_id: str, request_id: str) -> dict[str, bool]:
        fields(payload, set(), path="policy_close")
        with self._lock:
            self._sessions.pop(session_id, None)
        return {"closed": True}

    def smoke(self) -> None:
        task_id, request = synthetic_request(
            self.config.route.name,
            session_id="molmoact2-startup-smoke",
        )
        self.reset({"policy_seed": 1000, "task_id": task_id}, request.session_id, "startup-smoke-reset")
        result = self.act(request.to_mapping(), request.session_id, request.request_id)
        if result["execution_count"] != 1 or result["start_step"] != 0:
            raise RuntimeError("MolmoAct2 startup smoke returned an invalid action")
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
                actions, depth_cache = self._infer(encoded, session.generator, session.depth_cache)
                session.cache = session.adapter.select_native(actions)
                session.cache_index = 0
                session.depth_cache = depth_cache
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

    def _infer(
        self,
        payload: Mapping[str, Any],
        generator: Any,
        depth_cache: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray, Mapping[str, Any] | None]:
        images = tuple(payload["images"])
        if len(images) != 2:
            raise StrictSchemaError("policy_act.images: expected front and wrist RGB")
        for image in images:
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise StrictSchemaError("policy_act.images: expected uint8 HWC RGB")
        state = np.asarray(payload["state"])
        if state.dtype != np.float32 or state.shape != (8,) or not np.isfinite(state).all():
            raise StrictSchemaError("policy_act.state: expected finite float32[8]")
        config = self.config.value
        with self.torch.inference_mode():
            output = self.model.predict_action(
                processor=self.processor,
                images=list(images),
                task=string(payload["task"], "policy_act.task"),
                state=state,
                norm_tag=string(config["norm_tag"], "policy_config.norm_tag"),
                inference_action_mode=string(
                    config["inference_action_mode"],
                    "policy_config.inference_action_mode",
                ),
                enable_depth_reasoning=boolean(
                    config["enable_depth_reasoning"],
                    "policy_config.enable_depth_reasoning",
                ),
                enable_adaptive_depth=boolean(
                    config["enable_adaptive_depth"],
                    "policy_config.enable_adaptive_depth",
                ),
                depth_cache=depth_cache,
                num_steps=integer(config["num_steps"], "policy_config.num_steps", minimum=1),
                n_action_steps=integer(
                    config["execution_count"],
                    "policy_config.execution_count",
                    minimum=1,
                ),
                generator=generator,
                normalize_language=boolean(
                    config["normalize_language"],
                    "policy_config.normalize_language",
                ),
                enable_cuda_graph=boolean(
                    config["enable_cuda_graph"],
                    "policy_config.enable_cuda_graph",
                ),
            )
        actions = output.actions.detach().to("cpu", dtype=self.torch.float32).numpy()
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"MolmoAct2 returned invalid action shape {actions.shape}")
        return np.ascontiguousarray(actions, dtype=np.float32), output.depth_cache
