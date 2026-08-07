from __future__ import annotations

import json
from pathlib import Path

from robot_auto_evolve.runtime_paths import RuntimePaths, project_root_from_package


LIBERO_PRO_SOURCE_COMMIT = "eafdb809426b13153aa1e4c42d6601844217dfec"
LIBERO_PRO_ASSET_REPOSITORY = "zhouxueyang/LIBERO-Pro"
LIBERO_PRO_ASSET_REVISION = "3a33dc4cfbea7ff5eac6764e9c557500972b6259"
def default_asset_root(project_root: str | Path) -> Path:
    return RuntimePaths.load(project_root).artifact("libero_pro_assets")


def validate_libero_pro_assets(root: str | Path) -> dict[str, object]:
    assets = Path(root).resolve()
    suites = []
    for base_suite in ("spatial", "object", "goal", "10"):
        for perturbation in ("lan", "object", "swap", "task"):
            name = f"libero_{base_suite}_{perturbation}"
            bddl = tuple(sorted((assets / "bddl_files" / name).glob("*.bddl")))
            initial = tuple(sorted((assets / "init_files" / name).glob("*.pruned_init")))
            if len(bddl) != 10 or len(initial) != 10:
                raise RuntimeError(f"LIBERO-Pro suite assets are incomplete: {name}")
            if {path.stem for path in bddl} != {path.name.removesuffix(".pruned_init") for path in initial}:
                raise RuntimeError(f"LIBERO-Pro BDDL and initial-state tasks differ: {name}")
            suites.append(name)
    if any(path for group in ("bddl_files", "init_files") for path in (assets / group).glob("libero_*_env")):
        raise RuntimeError("unexpected LIBERO-Pro environment-perturbation assets")
    return {
        "repository": LIBERO_PRO_ASSET_REPOSITORY,
        "revision": LIBERO_PRO_ASSET_REVISION,
        "available_suites": suites,
        "missing_public_family": "env",
    }


def validate_libero_pro_source(root: str | Path) -> Path:
    source = Path(root).resolve()
    required = (
        source / "libero" / "libero" / "benchmark" / "__init__.py",
        source / "libero" / "libero" / "envs" / "__init__.py",
        source / ".git",
    )
    if any(not path.exists() for path in required):
        raise RuntimeError("LIBERO-Pro source checkout is incomplete")
    return source


def libero_pro_config_paths(source_root: str | Path, asset_root: str | Path) -> dict[str, str]:
    source = validate_libero_pro_source(source_root)
    assets = Path(asset_root).resolve()
    validate_libero_pro_assets(assets)
    package = source / "libero" / "libero"
    return {
        "assets": str(package / "assets"),
        "bddl_files": str(assets / "bddl_files"),
        "benchmark_root": str(package),
        "datasets": str(RuntimePaths.load(project_root_from_package()).artifact("libero_pro_datasets")),
        "init_states": str(assets / "init_files"),
    }


def write_libero_pro_config(source_root: str | Path, asset_root: str | Path, config_dir: str | Path) -> dict[str, str]:
    config = Path(config_dir).resolve()
    paths = libero_pro_config_paths(source_root, asset_root)
    if any(not Path(paths[name]).is_dir() for name in ("assets", "bddl_files", "init_states")):
        raise RuntimeError("LIBERO-Pro source or assets are incomplete")
    config.mkdir(parents=True, exist_ok=False)
    Path(paths["datasets"]).mkdir(parents=True, exist_ok=True)
    (config / "config.yaml").write_text(json.dumps(paths, sort_keys=True) + "\n", encoding="utf-8")
    return paths
