from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from robot_auto_evolve.services import MsgpackServiceServer, ServiceIdentity, serialize_methods

from .config import PolicyServiceConfig
from .rldx import RLDXRoboCasa365PolicyBackend


def _verify_gpu(gpu_id: int, device: str, expected_uuid: str) -> None:
    if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu_id):
        raise RuntimeError("RLDX replica requires CUDA_VISIBLE_DEVICES=<gpu-id> and --device cuda:0")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "--id", str(gpu_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != expected_uuid:
        raise RuntimeError("physical GPU UUID verification failed")


def main(argv: list[str] | None = None) -> int:
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
    if args.gpu_id < 0:
        parser.error("--gpu-id must be nonnegative")
    if "CONDA_PREFIX" not in os.environ or "VIRTUAL_ENV" in os.environ:
        raise RuntimeError("RLDX services require a dedicated conda environment")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("RLDX services require PYTHONNOUSERSITE=1")
    _verify_gpu(args.gpu_id, args.device, args.device_uuid)
    config = PolicyServiceConfig.load(args.config)
    backend = RLDXRoboCasa365PolicyBackend(config, args.source_root, args.device)
    backend.smoke()
    identity = ServiceIdentity(
        service_name="rldx_robocasa365",
        service_kind="policy",
        service_version="rldx-ebbfb4f6",
        protocol_version=1,
        model_id="RLWRLD/RLDX-1-FT-RC365",
        checkpoint_revision="587e9ecdcc5e7184fcc17f58713908edff5af041",
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
