from __future__ import annotations

from pathlib import Path

import yaml


def write_robocerebra_config(source_root: str | Path, output_dir: str | Path) -> Path:
    source = Path(source_root).resolve()
    package = source / "LIBERO" / "libero" / "libero"
    if not (package / "assets").is_dir() or not (package / "bddl_files").is_dir():
        raise RuntimeError("RoboCerebra LIBERO package is incomplete")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    value = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "datasets": str(package),
        "assets": str(package / "assets"),
    }
    path = output / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")
    return path
