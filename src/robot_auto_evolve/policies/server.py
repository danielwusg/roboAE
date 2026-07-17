from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from robot_auto_evolve.services import MsgpackServiceServer, ServiceIdentity, serialize_methods
from robot_auto_evolve.runtime_paths import assert_clean_import_origin, project_root_from_package

from .config import PolicyServiceConfig
from .molmoact2 import MolmoAct2LiberoPolicyBackend
from .molmoact2_droid import MolmoAct2DroidPolicyBackend
from .molmobot import MolmoBotDroidPolicyBackend
from .openvla import OpenVLAPolicyBackend
from .openpi_droid import OpenPiDroidJointPositionPolicyBackend
from .pi05 import Pi05LiberoPolicyBackend
from .rldx import RLDXRoboCasa365PolicyBackend
from .rlinf_pi05 import RLinfPi05LiberoPolicyBackend
from .smolvla import SmolVLARoboCerebraPolicyBackend
from .routes import UnavailablePolicyRoute
from .xvla import XVLAPolicyBackend


def build_backend(config: PolicyServiceConfig, source_root: Path, device: str):
    config.route.require_available()
    if config.route.backend == "xvla":
        return XVLAPolicyBackend(config, source_root, device)
    if config.route.backend == "openvla":
        return OpenVLAPolicyBackend(config, source_root, device)
    if config.route.backend == "pi05":
        return Pi05LiberoPolicyBackend(config, source_root, device)
    if config.route.backend == "smolvla":
        return SmolVLARoboCerebraPolicyBackend(config, source_root, device)
    if config.route.backend == "rlinf_pi05":
        return RLinfPi05LiberoPolicyBackend(config, source_root, device)
    if config.route.backend == "molmoact2":
        return MolmoAct2LiberoPolicyBackend(config, source_root, device)
    if config.route.backend == "molmoact2_droid":
        return MolmoAct2DroidPolicyBackend(config, source_root, device)
    if config.route.backend == "rldx":
        return RLDXRoboCasa365PolicyBackend(config, source_root, device)
    if config.route.backend == "molmobot":
        return MolmoBotDroidPolicyBackend(config, source_root, device)
    if config.route.backend == "openpi_droid_jointpos":
        return OpenPiDroidJointPositionPolicyBackend(config, source_root, device)
    raise UnavailablePolicyRoute(
        f"{config.route.name}: pinned upstream adapter exists, but this in-process backend has not passed GPU integration"
    )


def _verify_gpu(gpu_id: int, device: str, expected_uuid: str) -> None:
    if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu_id):
        raise RuntimeError("replica requires CUDA_VISIBLE_DEVICES=<gpu-id> and --device cuda:0")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "--id", str(gpu_id)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("physical GPU UUID verification timed out") from exc
    if result.returncode != 0 or result.stdout.strip() != expected_uuid:
        raise RuntimeError("physical GPU UUID verification failed")


def main(argv: list[str] | None = None) -> int:
    assert_clean_import_origin(project_root_from_package())
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--replica-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    if "CONDA_PREFIX" not in os.environ or "VIRTUAL_ENV" in os.environ:
        raise RuntimeError("policy services require a dedicated conda environment")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("policy services require PYTHONNOUSERSITE=1")
    _verify_gpu(args.gpu_id, args.device, args.device_uuid)
    config = PolicyServiceConfig.load(args.config)
    backend = build_backend(config, args.source_root, args.device)
    backend.smoke()
    identity = ServiceIdentity(
        service_name=config.route.name,
        service_kind="policy",
        service_version=f"{config.route.backend}-{config.route.source_commit[:8]}",
        protocol_version=1,
        model_id=config.route.model_id,
        checkpoint_revision=config.route.revision,
        config_sha256=config.sha256,
        stateful=True,
        replica_id=args.replica_id,
        gpu_ids=(args.gpu_id,),
    )
    server = MsgpackServiceServer(
        identity,
        serialize_methods({"act": backend.act, "reset": backend.reset, "close_session": backend.close_session}),
        host=args.host,
        port=args.port,
    )
    print(json.dumps({"ready": True, "identity": identity.to_mapping()}, sort_keys=True), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
