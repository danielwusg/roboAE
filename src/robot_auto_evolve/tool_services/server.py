from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from robot_auto_evolve.services import MsgpackServiceServer, ServiceIdentity, serialize_methods
from robot_auto_evolve.runtime_paths import assert_clean_import_origin, project_root_from_package

from .backends import (
    FixtureBackend,
    GroundingDinoBackend,
    OpenAICompatibleBackend,
    RoboPointBackend,
    Sam3Backend,
    ToolBackend,
    TransformersLanguageBackend,
    TransformersVisionBackend,
    api_key_from_environment,
)
from .graspgen import GraspGenBackend, semantic_config as graspgen_semantic_config
from .identities import MODEL_SPECS, config_hash, identity_for
from .molmo2 import Molmo2Backend


METHODS = {
    "language": ("generate",),
    "vision": ("describe",),
    "detection": ("detect",),
    "segmentation": ("segment",),
    "pointing": ("point",),
    "grasp": ("grasp",),
}

_NVIDIA_SMI_TIMEOUT_SECONDS = 5.0


def _load_identity(path: Path) -> ServiceIdentity:
    return ServiceIdentity.from_mapping(json.loads(path.read_text()))


def _runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    device_type = args.device.split(":", 1)[0]
    result = {
        "service": args.service,
        "runtime": args.runtime,
        "device_type": device_type,
        "upstream_model": args.model,
        "factory": args.factory,
    }
    if args.runtime == "openai-compatible":
        result["upstream_timeout_s"] = float(args.upstream_timeout)
    if args.runtime == "fixture":
        if args.fixture is None or not args.fixture.is_file():
            raise ValueError("--fixture is required for fixture runtime")
        import hashlib

        result["fixture_sha256"] = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    if args.service == "sam3":
        result.update(
            {
                "checkpoint_loader": "safetensors",
                "image_resolution": Sam3Backend.MODEL_RESOLUTION,
                "mask_threshold": 0.0,
                "max_hole_area": 256.0,
                "max_sprinkle_area": 0.0,
                "mask_selection": "maximum_predicted_iou",
                "multimask_output": True,
                "prompt_mode": "one_mask_per_box_or_point_set",
                "prompt_roundoff_tolerance_px": Sam3Backend.PROMPT_ROUNDOFF_TOLERANCE_PX,
                "safetensors_version": Sam3Backend.SAFETENSORS_VERSION,
            }
        )
    if args.service in {"molmo2_vision", "molmo2_pointing"}:
        result.update(
            {
                "checkpoint_loader": "exact_local_snapshot",
                "local_files_only": True,
                "processor": "apply_chat_template_tokenized",
                "remote_code_origin": "checkpoint_snapshot",
                "torch_dtype": Molmo2Backend.TORCH_DTYPE,
                "decoding": "greedy",
                "transformers_version": Molmo2Backend.TRANSFORMERS_VERSION,
            }
        )
        if args.service == "molmo2_pointing":
            result.update(
                {
                    "point_format": "html_coords_1000",
                    "point_max_tokens": Molmo2Backend.POINT_MAX_TOKENS,
                    "point_confidence": Molmo2Backend.POINT_CONFIDENCE,
                }
            )
    if args.service == "graspgen":
        result.update(graspgen_semantic_config())
    return result


def _identity(args: argparse.Namespace) -> ServiceIdentity:
    if args.identity_json is not None:
        identity = _load_identity(args.identity_json)
    else:
        identity = identity_for(
            args.service,
            gpu_id=args.gpu_id,
            replica_id=args.replica_id,
            runtime_config=_runtime_config(args),
            model_id=args.model_id,
            checkpoint_revision=args.checkpoint_revision,
        )
    spec = MODEL_SPECS[args.service]
    if identity.service_name != spec.service_name or identity.service_kind != spec.capability:
        raise ValueError("service identity does not match selected service")
    if identity.stateful:
        raise ValueError("tool service identity must declare stateful=false")
    if identity.gpu_id != args.gpu_id:
        raise ValueError("service identity gpu_id differs from --gpu-id")
    expected_model = args.model_id or spec.model_id
    expected_revision = args.checkpoint_revision or spec.checkpoint_revision
    if identity.model_id != expected_model or identity.checkpoint_revision != expected_revision:
        raise ValueError("service identity model or checkpoint differs from runtime")
    if identity.config_sha256 != config_hash(_runtime_config(args)):
        raise ValueError("service identity configuration hash differs from runtime")
    return identity


def _verify_device_uuid(args: argparse.Namespace) -> None:
    if not args.device.startswith("cuda"):
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "--id", str(args.gpu_id)],
            check=False,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("GPU UUID query exceeded five seconds") from exc
    if result.returncode != 0:
        raise RuntimeError("failed to query GPU UUID")
    actual = result.stdout.strip()
    if actual != args.device_uuid:
        raise RuntimeError(f"GPU UUID mismatch: got {actual!r}, expected {args.device_uuid!r}")


def _enforce_launch_status(status: str, reason: str | None, enable_pending_source_test: bool) -> None:
    if status in {"backend_implemented", "gpu_smoke_verified"}:
        return
    if status == "unavailable_pending_source_test" and enable_pending_source_test:
        return
    if reason is None:
        raise RuntimeError(f"tool service status is not launchable: {status}")
    raise RuntimeError(reason)


def _backend(args: argparse.Namespace, identity: ServiceIdentity) -> ToolBackend:
    spec = MODEL_SPECS[args.service]
    _enforce_launch_status(spec.status, spec.reason, args.enable_pending_source_test)
    if args.runtime == "fixture":
        if args.fixture is None:
            raise ValueError("--fixture is required for fixture runtime")
        return FixtureBackend(identity, args.fixture)
    if args.runtime == "openai-compatible":
        if args.base_url is None or args.model is None:
            raise ValueError("--base-url and --model are required")
        return OpenAICompatibleBackend(
            identity,
            args.base_url,
            args.model,
            api_key_from_environment(args.api_key_env),
            args.upstream_timeout,
        )
    if args.service == "qwen_language":
        return TransformersLanguageBackend(identity, args.device)
    if args.service == "qwen_vision":
        return TransformersVisionBackend(identity, args.device)
    if args.service in {"molmo2_vision", "molmo2_pointing"}:
        return Molmo2Backend(identity, args.device)
    if args.service == "grounding_dino":
        return GroundingDinoBackend(identity, args.device)
    if args.service == "sam3":
        if spec.checkpoint_sha256 is None:
            raise RuntimeError("SAM3 checkpoint SHA-256 is missing")
        return Sam3Backend(identity, args.device, spec.checkpoint_sha256)
    if args.service == "robopoint":
        return RoboPointBackend(identity, args.device)
    if args.service == "graspgen":
        if args.runtime != "official":
            raise ValueError("GraspGen requires the official runtime")
        return GraspGenBackend(identity, args.device)
    raise ValueError(f"unsupported service/runtime pair: {args.service}/{args.runtime}")


def make_server(host: str, port: int, backend: ToolBackend) -> MsgpackServiceServer:
    methods = {
        operation: (lambda payload, session_id, request_id, name=operation: backend.invoke(name, payload))
        for operation in METHODS[backend.identity.service_kind]
    }
    # The transformers / official backends each hold ONE GPU model in-process and are not thread-safe
    # for concurrent generate(), so their methods are serialized (one request at a time -- this is the
    # ~8-worker/GPU throughput ceiling that --vllm exists to lift). The OpenAICompatibleBackend is only
    # a stateless HTTP forwarder to a vLLM OpenAI server that does its OWN continuous batching, so
    # serializing it here would funnel all worker threads through one lock and defeat the entire point
    # of --vllm. Leave the proxy UNSERIALIZED so the worker threads reach vLLM concurrently and it can
    # batch them (the proxy holds no mutable state and requests.post is thread-safe).
    if not isinstance(backend, OpenAICompatibleBackend):
        methods = serialize_methods(methods)
    return MsgpackServiceServer(backend.identity, methods, host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    assert_clean_import_origin(project_root_from_package())
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument(
        "--runtime",
        choices=("transformers", "openai-compatible", "official", "fixture"),
        default="transformers",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--replica-id", required=True)
    parser.add_argument("--identity-json", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key-env")
    parser.add_argument("--upstream-timeout", type=float, default=120.0)
    parser.add_argument("--factory")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--enable-pending-source-test", action="store_true")
    args = parser.parse_args(argv)
    _verify_device_uuid(args)
    identity = _identity(args)
    backend = _backend(args, identity)
    backend.load()
    backend.smoke()
    server = make_server(args.host, args.port, backend)
    print(
        json.dumps(
            {
                "ready": True,
                "identity": identity.to_mapping(),
                "device_manifest": {
                    "device": args.device,
                    "logical_gpu_id": args.gpu_id,
                    "physical_device_uuid": args.device_uuid,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
