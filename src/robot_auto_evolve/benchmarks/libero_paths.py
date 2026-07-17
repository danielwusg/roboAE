from __future__ import annotations

import json
from pathlib import Path

from robot_auto_evolve.runtime_paths import RuntimePaths, project_root_from_package


def libero_shared_datasets(source_root: Path) -> Path:
    del source_root
    return RuntimePaths.load(project_root_from_package()).artifact("libero_datasets")


def libero_config_paths(source_root: Path) -> dict[str, str]:
    source = Path(source_root).resolve()
    package = source / "libero" / "libero"
    return {
        "assets": str(package / "assets"),
        "bddl_files": str(package / "bddl_files"),
        "benchmark_root": str(package),
        "datasets": str(libero_shared_datasets(source)),
        "init_states": str(package / "init_files"),
    }


def write_libero_config(source_root: Path, config_dir: Path) -> dict[str, str]:
    config = Path(config_dir).resolve()
    paths = libero_config_paths(source_root)
    if any(not Path(paths[name]).is_dir() for name in ("assets", "bddl_files", "init_states")):
        raise RuntimeError("pinned LIBERO checkout is incomplete")
    config.mkdir(parents=True, exist_ok=False)
    Path(paths["datasets"]).mkdir(parents=True, exist_ok=True)
    (config / "config.yaml").write_text(json.dumps(paths, sort_keys=True) + "\n", encoding="utf-8")
    return paths
