from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_auto_evolve.agent.api import VLARequest
from robot_auto_evolve.benchmarks.contracts import action_chunk
from robot_auto_evolve.benchmarks.libero_pro import split_task_id
from robot_auto_evolve.benchmarks.robocerebra import parse_task_id as parse_robocerebra_task_id
from robot_auto_evolve.benchmarks.libero_suites import LIBERO_TASK_SUITE
from robot_auto_evolve.benchmarks.rlinf_pi05 import EXECUTION_HORIZON, MODEL_HORIZON, RLinfPi05LiberoAdapter
from robot_auto_evolve.protocol.schema import StrictSchemaError, fields, integer, mapping, string

from .config import PolicyServiceConfig
from .smoke import synthetic_request
from .xvla import deterministic_seed


RLINF_OPENPI_COMMIT = "c5dc4b9296a1a4739bf52828f28a579f12dce763"
RLINF_LEROBOT_COMMIT = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
MODEL_SIZE_BYTES = 7_473_091_464
MODEL_SHA256 = "4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683"
METADATA_SIZE_BYTES = 3_236
METADATA_SHA256 = "8eba92f1159903755c6f529320913033a142a691bb148f90a96db0b6b91d6528"
NORM_STATS_SIZE_BYTES = 4_507
NORM_STATS_SHA256 = "dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1"
TOKENIZER_ID = "RLinf/openpi_tokenizer"
TOKENIZER_REVISION = "befaa248e4f82954b625a421658f933dfd1a97a0"
TOKENIZER_PATH = "big_vision/paligemma_tokenizer.model"
TOKENIZER_SIZE_BYTES = 4_264_023
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
TIED_EMBEDDING_ALIAS = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
TIED_EMBEDDING_TARGET = "paligemma_with_expert.paligemma.lm_head.weight"
COMPILE_MODE = "max-autotune-no-cudagraphs"
COMPILE_MODE_OPTIONS = {"coordinate_descent_tuning": True, "max_autotune": True}
_SAFETENSORS_DTYPES = {"BF16": "bfloat16", "F32": "float32"}


@dataclass
class _Session:
    policy_seed: int
    task_id: str
    adapter: RLinfPi05LiberoAdapter
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


def _verify_file(path: Path, size_bytes: int, sha256: str, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is absent")


def materialize_openpi_tokenizer(snapshot: Path, cache_root: Path) -> Path:
    source = snapshot / TOKENIZER_PATH
    _verify_file(source, TOKENIZER_SIZE_BYTES, TOKENIZER_SHA256, "OpenPI tokenizer source")
    target = cache_root / TOKENIZER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    _verify_file(target, TOKENIZER_SIZE_BYTES, TOKENIZER_SHA256, "OpenPI tokenizer cache")
    return cache_root.resolve()


def validate_checkpoint_schema(model_state: Mapping[str, Any], checkpoint_state: Mapping[str, tuple[tuple[int, ...], str]]) -> None:
    model_names = set(model_state)
    checkpoint_names = set(checkpoint_state)
    if model_names - checkpoint_names != {TIED_EMBEDDING_ALIAS} or checkpoint_names - model_names:
        raise RuntimeError("RLinf pi0.5 checkpoint parameter names differ")
    alias = model_state[TIED_EMBEDDING_ALIAS]
    target = model_state.get(TIED_EMBEDDING_TARGET)
    if target is None or alias.data_ptr() != target.data_ptr() or tuple(alias.shape) != tuple(target.shape):
        raise RuntimeError("RLinf pi0.5 checkpoint tied embedding alias differs")
    invalid = []
    for name in sorted(checkpoint_names):
        tensor = model_state[name]
        shape, dtype = checkpoint_state[name]
        expected_dtype = _SAFETENSORS_DTYPES.get(dtype)
        actual_dtype = str(tensor.dtype).removeprefix("torch.")
        if tuple(tensor.shape) != tuple(shape) or expected_dtype is None or actual_dtype != expected_dtype:
            invalid.append(name)
    if invalid:
        raise RuntimeError(f"RLinf pi0.5 checkpoint tensor metadata differs: {invalid[0]}")


def load_exact_pytorch_model(loader: Any, train_config: Any, weight_path: Path) -> Any:
    model = loader.pi0_pytorch.PI0Pytorch(config=train_config.model)
    with loader.safetensors.safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        checkpoint_state = {
            name: (tuple(handle.get_slice(name).get_shape()), handle.get_slice(name).get_dtype())
            for name in handle.keys()
        }
    validate_checkpoint_schema(model.state_dict(), checkpoint_state)
    missing, unexpected = loader.safetensors.torch.load_model(model, weight_path, strict=False)
    if set(missing) - {TIED_EMBEDDING_ALIAS} or unexpected:
        raise RuntimeError("RLinf pi0.5 checkpoint load result differs")
    return model


def validate_compile_mode(torch_module: Any, mode: str) -> None:
    inductor = getattr(torch_module, "_inductor", None)
    list_options = getattr(inductor, "list_mode_options", None)
    if mode != COMPILE_MODE or not callable(list_options):
        raise RuntimeError("RLinf pi0.5 compile mode differs")
    available = list_options()
    if not isinstance(available, Mapping) or available.get(mode) != COMPILE_MODE_OPTIONS:
        raise RuntimeError("RLinf pi0.5 compile mode options differ")
    config = getattr(inductor, "config", None)
    triton = getattr(config, "triton", None)
    if getattr(triton, "cudagraphs", None) is not False:
        raise RuntimeError("RLinf pi0.5 effective compile mode enables CUDA graphs")


def recompile_sample_actions(model: Any, torch_module: Any, mode: str) -> None:
    validate_compile_mode(torch_module, mode)
    wrapped = getattr(model, "sample_actions", None)
    original = getattr(wrapped, "_torchdynamo_orig_callable", None)
    bound = getattr(original, "__wrapped__", None)
    class_method = getattr(type(model), "sample_actions", None)
    try:
        captures = tuple(cell.cell_contents for cell in (getattr(original, "__closure__", None) or ()))
    except ValueError:
        captures = ()
    if (
        not callable(wrapped)
        or not callable(original)
        or not callable(bound)
        or getattr(bound, "__self__", None) is not model
        or getattr(bound, "__func__", None) is not class_method
        or len(captures) != 1
        or captures[0] is not bound
    ):
        raise RuntimeError("RLinf pi0.5 upstream compiled sample_actions wrapper differs")
    compiler = getattr(torch_module, "compile", None)
    if not callable(compiler):
        raise RuntimeError("RLinf pi0.5 torch compiler is absent")
    replacement = compiler(original, mode=mode)
    if not callable(replacement) or getattr(replacement, "_torchdynamo_orig_callable", None) is not original:
        raise RuntimeError("RLinf pi0.5 replacement compiled sample_actions wrapper differs")
    model.sample_actions = replacement


def create_exact_trained_policy(
    loader: Any,
    train_config: Any,
    checkpoint: Path,
    num_steps: int,
    device: str,
    torch_module: Any,
    compile_mode: str,
) -> Any:
    model = load_exact_pytorch_model(loader, train_config, checkpoint / "model.safetensors")
    recompile_sample_actions(model, torch_module, compile_mode)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None:
        raise RuntimeError("RLinf pi0.5 normalization asset id is absent")
    norm_stats = loader._checkpoints.load_norm_stats(checkpoint, data_config.asset_id)
    return loader._policy.Policy(
        model,
        transforms=[
            loader.transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            loader.transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            loader.transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        sample_kwargs={"num_steps": num_steps},
        metadata=train_config.policy_metadata,
        is_pytorch=True,
        pytorch_device=device,
    )


class RLinfPi05LiberoPolicyBackend:
    def __init__(self, config: PolicyServiceConfig, source_root: str | Path, device: str) -> None:
        if config.route.name not in {"rlinf_pi05_libero", "rlinf_pi05_libero_pro", "rlinf_pi05_robocerebra"}:
            raise StrictSchemaError("RLinf pi0.5 backend received a different route")
        configured_model_horizon = integer(config.value["action_horizon"], "policy_config.action_horizon")
        configured_execution_horizon = integer(config.value["execution_count"], "policy_config.execution_count")
        configured_denoise_steps = integer(config.value["denoise_steps"], "policy_config.denoise_steps")
        configured_compile_mode = string(config.value["compile_mode"], "policy_config.compile_mode")
        if configured_model_horizon != MODEL_HORIZON or configured_execution_horizon != EXECUTION_HORIZON:
            raise RuntimeError("RLinf pi0.5 configured action horizons differ")
        if configured_denoise_steps != 10:
            raise RuntimeError("RLinf pi0.5 configured denoise steps differ")
        self.config = config
        source = Path(source_root).resolve()
        openpi_source = source.parent / "rlinf_openpi"
        lerobot_source = source.parent / "rlinf_lerobot"
        if not (source / "rlinf" / "models" / "embodiment" / "openpi").is_dir() or _head(source) != config.route.source_commit:
            raise RuntimeError("RLinf source checkout mismatch")
        if not (openpi_source / "src" / "openpi" / "models_pytorch").is_dir() or _head(openpi_source) != RLINF_OPENPI_COMMIT:
            raise RuntimeError("RLinf OpenPI source checkout mismatch")
        if not (lerobot_source / "lerobot" / "common" / "datasets").is_dir() or _head(lerobot_source) != RLINF_LEROBOT_COMMIT:
            raise RuntimeError("RLinf LeRobot source checkout mismatch")
        _require_clean(source, "RLinf")
        _require_clean(openpi_source, "RLinf OpenPI")
        _require_clean(lerobot_source, "RLinf LeRobot")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        torch = importlib.import_module("torch")
        hub = importlib.import_module("huggingface_hub")
        openpi = importlib.import_module("openpi")
        lerobot = importlib.import_module("lerobot")
        if not _inside(openpi, openpi_source):
            raise RuntimeError("imported OpenPI package differs from the pinned RLinf fork")
        if not _inside(lerobot, lerobot_source):
            raise RuntimeError("imported LeRobot package differs from the pinned RLinf commit")
        if device != "cuda:0" or torch.cuda.device_count() != 1:
            raise RuntimeError("RLinf pi0.5 replica requires exactly one visible GPU as cuda:0")
        validate_compile_mode(torch, configured_compile_mode)
        checkpoint = mapping(config.value["checkpoint"], "policy_config.checkpoint")
        model_id = string(checkpoint["id"], "policy_config.checkpoint.id")
        revision = string(checkpoint["revision"], "policy_config.checkpoint.revision")
        snapshot = Path(
            hub.snapshot_download(
                repo_id=model_id,
                revision=revision,
                allow_patterns=["model.safetensors", "metadata.pt", "physical-intelligence/libero/norm_stats.json"],
                local_files_only=True,
            )
        ).resolve()
        tokenizer = Path(
            hub.snapshot_download(
                repo_id=TOKENIZER_ID,
                revision=TOKENIZER_REVISION,
                allow_patterns=[TOKENIZER_PATH],
                local_files_only=True,
            )
        ).resolve()
        if snapshot.name != revision or tokenizer.name != TOKENIZER_REVISION:
            raise RuntimeError("RLinf pi0.5 cached artifact revision mismatch")
        _verify_file(snapshot / "model.safetensors", MODEL_SIZE_BYTES, MODEL_SHA256, "RLinf pi0.5 model")
        _verify_file(snapshot / "metadata.pt", METADATA_SIZE_BYTES, METADATA_SHA256, "RLinf pi0.5 metadata")
        _verify_file(
            snapshot / "physical-intelligence" / "libero" / "norm_stats.json",
            NORM_STATS_SIZE_BYTES,
            NORM_STATS_SHA256,
            "RLinf pi0.5 normalization statistics",
        )
        _verify_file(tokenizer / TOKENIZER_PATH, TOKENIZER_SIZE_BYTES, TOKENIZER_SHA256, "OpenPI tokenizer")
        metadata = mapping(
            torch.load(snapshot / "metadata.pt", map_location="cpu", weights_only=False),
            "rlinf_pi05.metadata",
        )
        training = mapping(metadata["config"], "rlinf_pi05.metadata.config")
        checkpoint_model = mapping(training["model"], "rlinf_pi05.metadata.config.model")
        if (
            checkpoint_model.get("pi05") is not True
            or integer(checkpoint_model["action_horizon"], "rlinf_pi05.metadata.action_horizon") != MODEL_HORIZON
            or integer(checkpoint_model["action_chunk"], "rlinf_pi05.metadata.action_chunk") != EXECUTION_HORIZON
            or integer(checkpoint_model["num_steps"], "rlinf_pi05.metadata.num_steps") != configured_denoise_steps
            or integer(checkpoint_model["action_env_dim"], "rlinf_pi05.metadata.action_env_dim") != 7
            or string(checkpoint_model["simulator"], "rlinf_pi05.metadata.simulator") != "libero"
        ):
            raise RuntimeError("RLinf pi0.5 checkpoint metadata differs")
        runtime_tmp = os.environ.get("TMPDIR")
        if runtime_tmp is None:
            raise RuntimeError("RLinf pi0.5 runtime requires TMPDIR")
        openpi_data = materialize_openpi_tokenizer(tokenizer, Path(runtime_tmp).resolve() / "openpi")
        os.environ["OPENPI_DATA_HOME"] = str(openpi_data)
        dataconfig = importlib.import_module("rlinf.models.embodiment.openpi.dataconfig")
        loader = importlib.import_module("toolkits.standalone_eval_scripts.openpi")
        if not _inside(dataconfig, source) or not _inside(loader, source):
            raise RuntimeError("imported RLinf loader differs from pinned source")
        train_config = dataconfig._CONFIGS_DICT["pi05_libero"]
        model_config = train_config.model
        if (
            model_config.pi05 is not True
            or model_config.action_horizon != MODEL_HORIZON
            or model_config.discrete_state_input is not False
        ):
            raise RuntimeError("RLinf pi0.5 upstream model config differs")
        data_config = train_config.data.create(train_config.assets_dirs, model_config)
        if data_config.asset_id != "physical-intelligence/libero" or not data_config.use_quantile_norm:
            raise RuntimeError("RLinf pi0.5 upstream data config differs")
        self.policy = create_exact_trained_policy(
            loader,
            train_config,
            snapshot,
            configured_denoise_steps,
            device,
            torch,
            configured_compile_mode,
        )
        self.torch = torch
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def reset(self, payload: Any, session_id: str, request_id: str) -> dict[str, Any]:
        obj = fields(payload, {"policy_seed", "task_id"}, path="policy_reset")
        seed = integer(obj["policy_seed"], "policy_reset.policy_seed", minimum=0)
        task_id = string(obj["task_id"], "policy_reset.task_id")
        if self.config.route.name == "rlinf_pi05_libero":
            if task_id not in LIBERO_TASK_SUITE:
                raise StrictSchemaError("policy_reset.task_id: absent from standard LIBERO")
        elif self.config.route.name == "rlinf_pi05_robocerebra":
            parse_robocerebra_task_id(task_id)
        else:
            split_task_id(task_id)
        with self._lock:
            self._sessions[session_id] = _Session(seed, task_id, RLinfPi05LiberoAdapter())
        return {"policy_seed": seed, "task_id": task_id, "sample_index": 0}

    def close_session(self, payload: Any, session_id: str, request_id: str) -> dict[str, bool]:
        fields(payload, set(), path="policy_close")
        with self._lock:
            self._sessions.pop(session_id, None)
        return {"closed": True}

    def smoke(self) -> None:
        task_id, request = synthetic_request(self.config.route.name)
        self.reset({"policy_seed": 7, "task_id": task_id}, request.session_id, "startup-smoke-reset")
        result = self.act(request.to_mapping(), request.session_id, request.request_id)
        if result["execution_count"] != 1 or result["start_step"] != 0:
            raise RuntimeError("RLinf pi0.5 startup smoke returned an invalid action")
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
                encoded["prompt"] = request.instruction
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
        noise = np.random.default_rng(seed).standard_normal((MODEL_HORIZON, 32)).astype(np.float32)
        with self.torch.inference_mode():
            result = self.policy.infer(dict(payload), noise=noise)["actions"]
        actions = np.asarray(result, dtype=np.float32)
        if actions.shape != (MODEL_HORIZON, 7) or not np.isfinite(actions).all():
            raise RuntimeError(f"RLinf pi0.5 returned invalid action shape {actions.shape}")
        return np.ascontiguousarray(actions)
