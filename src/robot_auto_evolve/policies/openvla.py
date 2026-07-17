from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_auto_evolve.agent.api import VLARequest
from robot_auto_evolve.benchmarks.contracts import action_chunk
from robot_auto_evolve.benchmarks.openvla import (
    OPENVLA_GOOGLE_ACTION_SPEC,
    OPENVLA_GOOGLE_TASKS,
    decode_openvla_google_action,
)
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.protocol.schema import fields, integer, mapping, string
from robot_auto_evolve.runtime_paths import RuntimePaths, project_root_from_package

from .config import PolicyServiceConfig
from .smoke import synthetic_request


OPENVLA_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
OPENVLA_MODEL_ID = "openvla/openvla-7b"
SIMPLER_OPENVLA_REFERENCE_SHA256 = "74da205be0de0c86b4219d99393dc92fbf0e92fc2190bd0144ae4ce6c30cdc7b"
SIMPLER_OPENVLA_REFERENCE_FILE_COMMIT = "06b0cf23d3eb7f572c888993a042037336d1a52c"
OPENVLA_FILES = (
    ".gitattributes",
    "README.md",
    "added_tokens.json",
    "config.json",
    "configuration_prismatic.py",
    "generation_config.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "modeling_prismatic.py",
    "preprocessor_config.json",
    "processing_prismatic.py",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
OPENVLA_MINIMUM_BYTES = {
    "model-00001-of-00003.safetensors": 6_000_000_000,
    "model-00002-of-00003.safetensors": 6_000_000_000,
    "model-00003-of-00003.safetensors": 1_000_000_000,
    "model.safetensors.index.json": 90_000,
    "tokenizer.json": 1_000_000,
    "tokenizer.model": 400_000,
}


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    if algorithm == "sha1":
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_address(path: Path, blobs: Path, *, hash_files: bool) -> dict[str, Any]:
    if not path.is_symlink():
        raise RuntimeError(f"OpenVLA snapshot entry is not cache-linked: {path.name}")
    target = path.resolve(strict=True)
    try:
        target.relative_to(blobs)
    except ValueError as exc:
        raise RuntimeError(f"OpenVLA snapshot entry escapes its blob store: {path.name}") from exc
    address = target.name
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", address) is None:
        raise RuntimeError(f"OpenVLA blob has an invalid content address: {path.name}")
    algorithm = "sha256" if len(address) == 64 else "sha1"
    digest = _hash_file(target, algorithm) if hash_files else address
    if digest != address:
        raise RuntimeError(f"OpenVLA cached blob hash differs: {path.name}")
    return {
        "path": path.name,
        "blob": address,
        "hash_algorithm": algorithm,
        "sha256": _hash_file(target, "sha256") if hash_files and algorithm != "sha256" else digest,
        "size_bytes": target.stat().st_size,
    }


def openvla_snapshot(cache_root: str | Path | None = None, *, hash_files: bool = True) -> tuple[Path, list[dict[str, Any]]]:
    hub = Path(
        cache_root
        if cache_root is not None
        else os.environ.get(
            "HF_HUB_CACHE",
            str(RuntimePaths.load(project_root_from_package()).artifact("huggingface_hub")),
        )
    ).resolve()
    repository = hub / "models--openvla--openvla-7b"
    snapshot = repository / "snapshots" / OPENVLA_REVISION
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise RuntimeError(f"OpenVLA base snapshot is absent: {snapshot}")
    names = {path.name for path in snapshot.iterdir()}
    missing = sorted(set(OPENVLA_FILES) - names)
    unknown = sorted(names - set(OPENVLA_FILES))
    if missing or unknown:
        raise RuntimeError(f"OpenVLA base snapshot file set differs; missing={missing}, unknown={unknown}")
    records = []
    for name in OPENVLA_FILES:
        path = snapshot / name
        record = _content_address(path, repository / "blobs", hash_files=hash_files)
        if record["size_bytes"] < OPENVLA_MINIMUM_BYTES.get(name, 1):
            raise RuntimeError(f"OpenVLA cached file is truncated: {name}")
        records.append(record)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = set(mapping(index.get("weight_map"), "openvla_index.weight_map").values())
    expected_shards = {name for name in OPENVLA_FILES if name.endswith(".safetensors")}
    if shards != expected_shards:
        raise RuntimeError("OpenVLA index shard set differs")
    total_size = mapping(index.get("metadata"), "openvla_index.metadata").get("total_size")
    if type(total_size) is not int or total_size < 14_000_000_000:
        raise RuntimeError("OpenVLA index total size differs")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["OpenVLAForActionPrediction"]:
        raise RuntimeError("OpenVLA architecture identity differs")
    norm_stats = config.get("norm_stats")
    google_stats = None if not isinstance(norm_stats, dict) else norm_stats.get("fractal20220817_data")
    if not isinstance(google_stats, dict):
        raise RuntimeError("OpenVLA Google Robot normalization statistics are absent")
    action_stats = google_stats.get("action")
    if (
        not isinstance(action_stats, dict)
        or action_stats.get("mask") != [True, True, True, True, True, True, False]
        or any(not isinstance(action_stats.get(name), list) or len(action_stats[name]) != 7 for name in ("q01", "q99"))
    ):
        raise RuntimeError("OpenVLA Google Robot normalization statistics differ")
    return snapshot, records


def verified_openvla_runtime_assets(config: PolicyServiceConfig) -> tuple[Path, list[dict[str, Any]]]:
    manifest_path = RuntimePaths.load(project_root_from_package()).artifact("openvla_runtime_manifest")
    if not manifest_path.is_file():
        raise RuntimeError("OpenVLA full-hash setup manifest is absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_hashes = manifest.get("policy_config_sha256")
    checkpoint = manifest.get("checkpoint")
    if (
        manifest.get("complete") is not True
        or not isinstance(config_hashes, dict)
        or config_hashes.get(config.route.name) != config.sha256
        or manifest.get("source_commit") != config.route.source_commit
        or manifest.get("submodule_commit") != "cd45dd27dc6bb26d048cb6570cdab4e3f935cc37"
        or manifest.get("implementation_sha256") != config.value["reference_file_sha256"]
        or not isinstance(checkpoint, dict)
        or checkpoint.get("id") != config.route.model_id
        or checkpoint.get("revision") != config.route.revision
        or checkpoint.get("lock_name") != "openvla_base"
        or manifest.get("versions", {}).get("flash-attn") != "2.6.1"
    ):
        raise RuntimeError("OpenVLA full-hash setup manifest identity differs")
    snapshot, records = openvla_snapshot(hash_files=False)
    manifest_files = checkpoint.get("files")
    if not isinstance(manifest_files, list) or checkpoint.get("file_count") != len(records):
        raise RuntimeError("OpenVLA full-hash setup manifest file records differ")
    expected = {
        item["path"]: (item["blob"], item["hash_algorithm"], item["size_bytes"])
        for item in manifest_files
        if isinstance(item, dict) and {"path", "blob", "hash_algorithm", "size_bytes"} <= set(item)
    }
    actual = {
        item["path"]: (item["blob"], item["hash_algorithm"], item["size_bytes"])
        for item in records
    }
    if expected != actual or checkpoint.get("logical_size_bytes") != sum(item["size_bytes"] for item in records):
        raise RuntimeError("OpenVLA cached content addresses differ from the full-hash setup manifest")
    return snapshot, records


def _load_openvla(config: PolicyServiceConfig, snapshot: Path, transformers: Any, torch: Any, device: Any) -> tuple[Any, Any]:
    processor = transformers.AutoProcessor.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True
    )
    model = transformers.AutoModelForVision2Seq.from_pretrained(
        str(snapshot),
        attn_implementation=string(config.value["attn_implementation"], "policy_config.attn_implementation"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    return processor, model


@dataclass
class _Session:
    policy_seed: int
    task_id: str
    previous_gripper_action: float | None = None
    sticky_action_is_on: bool = False
    sticky_gripper_action: float = 0.0
    gripper_action_repeat: int = 0
    last_step: int = -1
    last_request_id: str | None = None
    last_response: dict[str, Any] | None = None
    instruction: str | None = None

    def prepare_instruction(self, instruction: str) -> None:
        if self.instruction is not None and instruction != self.instruction:
            self.previous_gripper_action = None
            self.sticky_action_is_on = False
            self.sticky_gripper_action = 0.0
            self.gripper_action_repeat = 0
        self.instruction = instruction

    def gripper(self, open_gripper: float, repeats: int) -> float:
        if not np.isfinite(open_gripper):
            raise StrictSchemaError("OpenVLA gripper prediction is non-finite")
        relative = 0.0 if self.previous_gripper_action is None else self.previous_gripper_action - open_gripper
        self.previous_gripper_action = open_gripper
        if abs(relative) > 0.5 and not self.sticky_action_is_on:
            self.sticky_action_is_on = True
            self.sticky_gripper_action = relative
        if self.sticky_action_is_on:
            self.gripper_action_repeat += 1
            relative = self.sticky_gripper_action
        if self.gripper_action_repeat == repeats:
            self.sticky_action_is_on = False
            self.sticky_gripper_action = 0.0
            self.gripper_action_repeat = 0
        return float(relative)


class OpenVLAPolicyBackend:
    def __init__(self, config: PolicyServiceConfig, source_root: str | Path, device: str) -> None:
        if config.route.backend != "openvla":
            raise StrictSchemaError("OpenVLA backend received a different route")
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("OpenVLA inference requires offline mode")
        if os.environ.get("TOKENIZERS_PARALLELISM") != "false":
            raise RuntimeError("OpenVLA inference requires TOKENIZERS_PARALLELISM=false")
        self.config = config
        self.source_root = Path(source_root).resolve()
        expected_source = (self.source_root / string(config.value["reference_file"], "policy_config.reference_file")).resolve()
        try:
            expected_source.relative_to(self.source_root)
        except ValueError as exc:
            raise RuntimeError("OpenVLA reference file escapes its source checkout") from exc
        if not expected_source.is_file() or _hash_file(expected_source, "sha256") != config.value["reference_file_sha256"]:
            raise RuntimeError("pinned SimplerEnv-OpenVLA reference implementation differs")
        head = subprocess.check_output(["git", "-C", str(self.source_root), "rev-parse", "HEAD"], text=True).strip()
        if head != config.route.source_commit:
            raise RuntimeError("OpenVLA harness source revision differs")
        dirty = subprocess.check_output(
            ["git", "-C", str(self.source_root), "status", "--porcelain=v1", "--untracked-files=all"],
            text=True,
        ).strip()
        if dirty:
            raise RuntimeError(f"SimplerEnv-OpenVLA reference source is dirty: {dirty.splitlines()[0]}")
        file_commit = subprocess.check_output(
            ["git", "-C", str(self.source_root), "log", "-1", "--format=%H", "--", str(expected_source.relative_to(self.source_root))],
            text=True,
        ).strip()
        if file_commit != config.value["reference_file_commit"]:
            raise RuntimeError("pinned SimplerEnv-OpenVLA reference file revision differs")
        snapshot, self.checkpoint_files = verified_openvla_runtime_assets(config)
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        if device != "cuda:0" or torch.cuda.device_count() != 1:
            raise RuntimeError("OpenVLA replica requires exactly one visible GPU as cuda:0")
        self.device = torch.device(device)
        self.processor, self.model = _load_openvla(config, snapshot, transformers, torch, self.device)
        if "fractal20220817_data" not in getattr(self.model, "norm_stats", {}):
            raise RuntimeError("loaded OpenVLA model lacks exact Google Robot normalization statistics")
        self.torch = torch
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def reset(self, payload: Any, session_id: str, request_id: str) -> dict[str, Any]:
        obj = fields(payload, {"policy_seed", "task_id"}, path="policy_reset")
        seed = integer(obj["policy_seed"], "policy_reset.policy_seed", minimum=0)
        task_id = string(obj["task_id"], "policy_reset.task_id")
        if task_id not in OPENVLA_GOOGLE_TASKS:
            raise StrictSchemaError("policy_reset.task_id: unsupported OpenVLA Google Robot task")
        with self._lock:
            self._sessions[session_id] = _Session(seed, task_id)
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
            raise RuntimeError("OpenVLA startup smoke returned an invalid action")
        self.close_session({}, request.session_id, "startup-smoke-close")

    def act(self, payload: Any, session_id: str, request_id: str) -> dict[str, Any]:
        request = VLARequest.from_mapping(payload)
        if request.session_id != session_id or request.request_id != request_id:
            raise StrictSchemaError("policy_act: payload and envelope identity mismatch")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise StrictSchemaError("policy session must be reset before act")
            if session.last_request_id == request_id:
                if session.last_response is None:
                    raise RuntimeError("OpenVLA idempotency record is incomplete")
                return session.last_response
            step = request.observation.step_index
            if step != session.last_step + 1:
                raise StrictSchemaError("OpenVLA observation steps must be consecutive")
            session.prepare_instruction(string(request.instruction, "policy_act.instruction"))
            raw = self._infer(request)
            gripper = session.gripper(float(raw[0, 6]), integer(self.config.value["sticky_gripper_steps"], "policy_config.sticky_gripper_steps"))
            converted = decode_openvla_google_action(raw, gripper)
            response = action_chunk(
                converted[None],
                spec=OPENVLA_GOOGLE_ACTION_SPEC,
                execution_count=1,
                request_id=request.request_id,
                session_id=request.session_id,
                start_step=step,
            ).to_mapping()
            session.last_step = step
            session.last_request_id = request_id
            session.last_response = response
            return response

    def _infer(self, request: VLARequest) -> np.ndarray:
        import cv2
        from PIL import Image

        try:
            rgb = request.observation.cameras["main"].rgb
        except KeyError as exc:
            raise StrictSchemaError("OpenVLA requires the main RGB camera") from exc
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise StrictSchemaError("OpenVLA main image must be uint8 HWC RGB")
        size = integer(self.config.value["image_size"], "policy_config.image_size", minimum=1)
        resized = cv2.resize(np.ascontiguousarray(rgb), (size, size), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(resized).convert("RGB")
        prompt = string(request.instruction, "policy_act.instruction")
        inputs = self.processor(prompt, image).to(self.device, dtype=self.torch.bfloat16)
        with self.torch.inference_mode():
            value = self.model.predict_action(
                **inputs,
                unnorm_key=string(self.config.value["unnorm_key"], "policy_config.unnorm_key"),
                do_sample=False,
            )
        result = np.asarray(value, dtype=np.float32).reshape(1, -1)
        if result.shape != (1, 7) or not np.isfinite(result).all():
            raise RuntimeError(f"OpenVLA returned invalid action shape {result.shape}")
        return np.ascontiguousarray(result)
