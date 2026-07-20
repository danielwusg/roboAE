from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_auto_evolve.agent.api import VLARequest
from robot_auto_evolve.benchmarks.contracts import action_chunk
from robot_auto_evolve.benchmarks.xvla import (
    CALVIN_TASKS,
    GOOGLE_VA_RULES,
    GOOGLE_VM_RULES,
    LIBERO_TASKS,
    ROBOTWIN_TASKS,
    VLABENCH_TASKS,
    WIDOWX_GRIPPER_THRESHOLDS,
    XVLACalvinAdapter,
    XVLAGoogleAdapter,
    XVLALiberoAdapter,
    XVLARoboTwinAdapter,
    XVLAVLABenchAdapter,
    XVLAWidowXAdapter,
)
from robot_auto_evolve.protocol.schema import StrictSchemaError, fields, integer, mapping, string

from .config import PolicyServiceConfig
from .smoke import synthetic_request


def deterministic_seed(policy_seed: int, call_index: int, resample_index: int = 0) -> int:
    payload = f"{policy_seed}\0{call_index}\0{resample_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


@dataclass
class _Session:
    policy_seed: int
    task_id: str
    adapter: Any
    call_index: int = 0
    initialized: bool = False
    cache_native: np.ndarray | None = None
    cache_values: np.ndarray | None = None
    cache_index: int = 0
    pending_native: np.ndarray | None = None
    pending_step: int | None = None
    last_request_id: str | None = None
    last_response: dict[str, Any] | None = None


def _adapter(route_name: str, task_id: str) -> Any:
    if route_name == "xvla_libero" and task_id in LIBERO_TASKS:
        return XVLALiberoAdapter()
    if route_name == "xvla_calvin" and task_id in CALVIN_TASKS:
        return XVLACalvinAdapter()
    if route_name == "xvla_simpler_widowx" and task_id in WIDOWX_GRIPPER_THRESHOLDS:
        return XVLAWidowXAdapter(task_id)
    if route_name == "xvla_simpler_google_va" and task_id in GOOGLE_VA_RULES:
        return XVLAGoogleAdapter(task_id, GOOGLE_VA_RULES)
    if route_name == "xvla_pt_simpler_google_va" and task_id in GOOGLE_VA_RULES:
        return XVLAGoogleAdapter(task_id, GOOGLE_VA_RULES)
    if route_name == "xvla_simpler_google_vm" and task_id in GOOGLE_VM_RULES:
        return XVLAGoogleAdapter(task_id, GOOGLE_VM_RULES)
    if route_name == "xvla_robotwin2" and task_id in ROBOTWIN_TASKS:
        return XVLARoboTwinAdapter()
    if route_name == "xvla_vlabench" and task_id in VLABENCH_TASKS:
        return XVLAVLABenchAdapter()
    raise StrictSchemaError(f"policy_reset.task_id: unsupported for route {route_name!r}")


def _resample_index(context: tuple[str, ...]) -> int:
    values = [item.removeprefix("policy_resample_index=") for item in context if item.startswith("policy_resample_index=")]
    if len(values) > 1 or (values and not values[0].isdigit()):
        raise StrictSchemaError("vla.context: invalid policy_resample_index")
    return int(values[0]) if values else 0


class XVLAPolicyBackend:
    def __init__(self, config: PolicyServiceConfig, source_root: str | Path, device: str) -> None:
        if config.route.backend != "xvla":
            raise StrictSchemaError("X-VLA backend received a different route")
        self.config = config
        self.source_root = Path(source_root).resolve()
        if not (self.source_root / "models" / "modeling_xvla.py").is_file():
            raise FileNotFoundError("X-VLA source checkout is incomplete")
        head = subprocess.check_output(
            ["git", "-C", str(self.source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != config.route.source_commit:
            raise RuntimeError(f"X-VLA source revision mismatch: {head}")
        dirty = subprocess.check_output(
            ["git", "-C", str(self.source_root), "status", "--porcelain=v1", "--untracked-files=all"],
            text=True,
        ).strip()
        if dirty:
            raise RuntimeError(f"X-VLA source working tree is dirty: {dirty.splitlines()[0]}")
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        torch = importlib.import_module("torch")
        hub = importlib.import_module("huggingface_hub")
        model_module = importlib.import_module("models.modeling_xvla")
        processor_module = importlib.import_module("models.processing_xvla")
        checkpoint = mapping(config.value["checkpoint"], "policy_config.checkpoint")
        model_id = string(checkpoint["id"], "policy_config.checkpoint.id")
        revision = string(checkpoint["revision"], "policy_config.checkpoint.revision")
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("X-VLA inference requires offline mode")
        snapshot = Path(
            hub.snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)
        ).resolve()
        if snapshot.name != revision:
            raise RuntimeError("X-VLA cached checkpoint revision mismatch")
        if device != "cuda:0" or torch.cuda.device_count() != 1:
            raise RuntimeError("X-VLA replica requires exactly one visible GPU as cuda:0")
        self.device = torch.device(device)
        self.model = model_module.XVLA.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to(self.device).to(torch.float32).eval()
        self.processor = processor_module.XVLAProcessor.from_pretrained(
            str(snapshot),
            local_files_only=True,
        )
        self.torch = torch
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def reset(self, payload: Any, session_id: str, request_id: str) -> dict[str, Any]:
        obj = fields(payload, {"policy_seed", "task_id"}, path="policy_reset")
        seed = integer(obj["policy_seed"], "policy_reset.policy_seed", minimum=0)
        task_id = string(obj["task_id"], "policy_reset.task_id")
        adapter = _adapter(self.config.route.name, task_id)
        with self._lock:
            self._sessions[session_id] = _Session(seed, task_id, adapter)
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
            raise RuntimeError("X-VLA startup smoke returned an invalid action")
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
                elif not request.refresh:
                    raise StrictSchemaError("policy_act: previous action is not observed as executed")
                elif step != session.pending_step:
                    raise StrictSchemaError("policy_act: previous action is not observed as executed")
                session.pending_native = None
                session.pending_step = None
            if request.refresh:
                session.cache_native = None
                session.cache_values = None
                session.cache_index = 0
            if session.cache_native is None or session.cache_index >= session.cache_native.shape[0]:
                encoded = session.adapter.encode(request.observation)
                encoded["language_instruction"] = request.instruction
                resample = _resample_index(request.context)
                seed = deterministic_seed(session.policy_seed, session.call_index, resample)
                native = self._infer(encoded, seed)
                selected = session.adapter.select_native(native)
                decoded = session.adapter.decode_selected(
                    selected,
                    request_id=request.request_id,
                    session_id=request.session_id,
                    start_step=step,
                )
                session.cache_native = selected
                session.cache_values = decoded.values
                session.cache_index = 0
                session.call_index += 1
            index = session.cache_index
            if session.cache_values is None or session.cache_native is None:
                raise RuntimeError("policy cache is incomplete")
            result = action_chunk(
                session.cache_values[index : index + 1],
                spec=session.adapter.action_spec,
                execution_count=1,
                request_id=request.request_id,
                session_id=request.session_id,
                start_step=step,
            )
            session.pending_native = session.cache_native[index].copy()
            session.pending_step = step
            session.cache_index += 1
            response = result.to_mapping()
            session.last_request_id = request_id
            session.last_response = response
        return response

    def _infer(self, payload: Mapping[str, Any], seed: int) -> np.ndarray:
        images = []
        for name in ("image0", "image1", "image2"):
            if name in payload:
                array = np.asarray(payload[name])
                if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                    raise StrictSchemaError(f"policy_act.{name}: expected uint8 HWC RGB")
                images.append(array)
        inputs = self.processor(images, string(payload["language_instruction"], "policy_act.language_instruction"))
        if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
            raise RuntimeError("X-VLA processor returned incomplete inputs")
        proprio = np.asarray(payload["proprio"])
        if proprio.dtype != np.float32 or proprio.shape != (20,):
            raise StrictSchemaError("policy_act.proprio: expected float32[20]")
        torch = self.torch
        dtype = next(self.model.parameters()).dtype

        def move(tensor: Any) -> Any:
            value = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(tensor)
            return value.to(self.device, dtype=dtype) if value.is_floating_point() else value.to(self.device)

        model_inputs = {name: move(value) for name, value in inputs.items()}
        model_inputs["proprio"] = move(proprio).unsqueeze(0)
        model_inputs["domain_id"] = torch.tensor([integer(payload["domain_id"], "policy_act.domain_id", minimum=0)], device=self.device)
        steps = integer(payload.get("steps", self.config.value["denoise_steps"]), "policy_act.steps", minimum=1)
        with torch.random.fork_rng(devices=[self.device.index]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            actions = self.model.generate_actions(**model_inputs, steps=steps)
        result = actions.squeeze(0).float().cpu().numpy().astype(np.float32, copy=False)
        if result.shape != (30, 20) or not np.isfinite(result).all():
            raise RuntimeError(f"X-VLA returned invalid action shape {result.shape}")
        return np.ascontiguousarray(result)
