from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


_PATH_KEYS = {
    "artifacts",
    "compile_cache_namespace",
    "environment_root",
    "schema_version",
    "shared_cache_root",
    "source_root",
}
_ARTIFACT_KEYS = {
    "compile_cache",
    "huggingface_hub",
    "libero_datasets",
    "libero_pro_assets",
    "libero_pro_datasets",
    "openvla_runtime_manifest",
    "robocasa365_asset_lock",
    "robocerebra_asset_lock",
    "robocerebra_assets",
    "simpler_xvla_source",
}


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"cannot read strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{name} fields differ; missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}")


def _absolute(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return Path(value).resolve()


def project_root_from_package() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    config_path: Path
    environment_root: Path
    source_root: Path
    shared_cache_root: Path
    compile_cache_namespace: str
    artifacts: Mapping[str, Path]

    @classmethod
    def load(cls, project_root: str | Path, config_path: str | Path | None = None) -> "RuntimePaths":
        root = Path(project_root).resolve()
        selected = config_path or os.environ.get("ROBOT_AE_RUNTIME_PATHS") or root / "runtime_paths.json"
        path = Path(selected).resolve()
        value = _json(path)
        _fields(value, _PATH_KEYS, "runtime_paths")
        if value["schema_version"] != 1:
            raise ValueError("runtime_paths.schema_version must be 1")
        artifacts = value["artifacts"]
        if not isinstance(artifacts, dict):
            raise ValueError("runtime_paths.artifacts must be an object")
        _fields(artifacts, _ARTIFACT_KEYS, "runtime_paths.artifacts")
        namespace = value["compile_cache_namespace"]
        if not isinstance(namespace, str) or not namespace or "/" in namespace or namespace in {".", ".."}:
            raise ValueError("runtime_paths.compile_cache_namespace is invalid")
        return cls(
            root,
            path,
            _absolute(value["environment_root"], "runtime_paths.environment_root"),
            _absolute(value["source_root"], "runtime_paths.source_root"),
            _absolute(value["shared_cache_root"], "runtime_paths.shared_cache_root"),
            namespace,
            {key: _absolute(item, f"runtime_paths.artifacts.{key}") for key, item in artifacts.items()},
        )

    def artifact(self, key: str) -> Path:
        try:
            return self.artifacts[key]
        except KeyError as exc:
            raise KeyError(f"unknown runtime artifact: {key}") from exc

    def environment_python(self, name: str) -> Path:
        if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
            raise ValueError("invalid environment name")
        path = (self.environment_root / name / "bin" / "python").resolve()
        if not path.is_relative_to(self.environment_root):
            raise ValueError("environment path escapes environment root")
        return path

    def source(self, directory: str) -> Path:
        if not isinstance(directory, str) or not directory:
            raise ValueError("invalid source directory")
        path = (self.source_root / directory).resolve()
        if not path.is_relative_to(self.source_root) or path == self.source_root:
            raise ValueError("source path escapes source root")
        return path


def clean_src(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    source = (root / "src").resolve()
    package = source / "robot_auto_evolve" / "__init__.py"
    if not package.is_file() or package.is_symlink():
        raise RuntimeError(f"clean package source is missing: {package}")
    return source


def assert_run_runtime_path(project_root: str | Path, value: str | Path) -> Path:
    root = Path(project_root).resolve()
    path = Path(value).resolve()
    try:
        relative = path.relative_to(root / "runs")
    except ValueError as exc:
        raise RuntimeError("runtime path must be below runs/<run-id>/runtime") from exc
    if len(relative.parts) < 2 or relative.parts[1] != "runtime":
        raise RuntimeError("runtime path must be below runs/<run-id>/runtime")
    return path


def clean_python_path(project_root: str | Path, *extra: str | Path) -> str:
    source = clean_src(project_root)
    items = [source]
    for value in extra:
        path = Path(value).resolve()
        foreign = path / "robot_auto_evolve" / "__init__.py"
        if foreign.exists() and path != source:
            raise RuntimeError(f"PYTHONPATH entry contains a foreign robot_auto_evolve package: {path}")
        if path not in items:
            items.append(path)
    return os.pathsep.join(map(str, items))


def clean_import_environment(project_root: str | Path, paths: RuntimePaths | None = None) -> dict[str, str]:
    root = Path(project_root).resolve()
    catalog = paths or RuntimePaths.load(root)
    if catalog.project_root != root:
        raise RuntimeError("runtime path catalog belongs to a different project root")
    source = clean_src(root)
    return {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(source),
        "ROBOT_AE_CLEAN_SRC": str(source),
        "ROBOT_AE_PROJECT_ROOT": str(root),
        "ROBOT_AE_RUNTIME_PATHS": str(catalog.config_path),
    }


def assert_clean_import_origin(project_root: str | Path, module: Any | None = None) -> Path:
    root = Path(project_root).resolve()
    if module is None:
        import robot_auto_evolve as module
    origin_value = getattr(module, "__file__", None)
    search_values = getattr(module, "__path__", ())
    if not isinstance(origin_value, str):
        raise RuntimeError("robot_auto_evolve import has no file origin")
    source = clean_src(root)
    expected = source / "robot_auto_evolve"
    origin = Path(origin_value).resolve()
    searches = tuple(Path(item).resolve() for item in search_values)
    if origin != expected / "__init__.py" or searches != (expected,):
        raise RuntimeError(f"robot_auto_evolve import escaped clean src: origin={origin}, search={searches}")
    return origin


def verify_python_import_origin(python: str | Path, project_root: str | Path, paths: RuntimePaths | None = None) -> Path:
    executable = Path(python).resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Python executable is missing: {executable}")
    root = Path(project_root).resolve()
    expected = clean_src(root) / "robot_auto_evolve" / "__init__.py"
    probe = (
        "import json,pathlib,robot_auto_evolve;"
        "print(json.dumps({'origin':str(pathlib.Path(robot_auto_evolve.__file__).resolve()),"
        "'search':[str(pathlib.Path(x).resolve()) for x in robot_auto_evolve.__path__]}))"
    )
    environment = {
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": f"{executable.parent}:/usr/bin:/bin",
        **clean_import_environment(root, paths),
    }
    result = subprocess.run(
        [str(executable), "-s", "-c", probe],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"clean import probe failed for {executable}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"clean import probe returned invalid output for {executable}") from exc
    expected_search = [str(expected.parent)]
    if payload != {"origin": str(expected), "search": expected_search}:
        raise RuntimeError(f"robot_auto_evolve import escaped clean src for {executable}: {payload}")
    return expected


ENVIRONMENT_NAMES = (
    "agent",
    "calvin",
    "core",
    "grounding_dino",
    "language",
    "libero",
    "libero_pro",
    "molmo2",
    "molmoact2",
    "openvla",
    "pi05",
    "rldx",
    "rlinf_pi05",
    "robocasa365",
    "robotwin2",
    "sam3",
    "simpler_openvla",
    "simpler_xvla",
    "vision",
    "vlabench",
    "xvla",
)


def _existing(paths: RuntimePaths) -> None:
    roots = (paths.environment_root, paths.source_root, paths.shared_cache_root)
    missing = [str(path) for path in (*roots, *paths.artifacts.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"declared runtime paths are missing: {missing}")


def verify_runtime(
    project_root: str | Path,
    *,
    verify_imports: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths = RuntimePaths.load(root)
    assert_clean_import_origin(root)
    _existing(paths)
    environments: list[str] = []
    imports: list[str] = []
    for name in ENVIRONMENT_NAMES:
        python = paths.environment_python(name)
        if not python.is_file():
            raise FileNotFoundError(f"declared conda environment is missing: {python}")
        environments.append(name)
        if verify_imports:
            verify_python_import_origin(python, root, paths)
            imports.append(name)
    return {
        "complete": True,
        "project_root": str(root),
        "runtime_paths": str(paths.config_path),
        "verified": {
            "environment_import_origins": imports,
            "environments": environments,
        },
        "declared_only": {
            "artifacts": sorted(paths.artifacts),
            "roots": ["environment_root", "shared_cache_root", "source_root"],
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m robot_auto_evolve.runtime_paths")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--project-root", type=Path, default=project_root_from_package())
    verify.add_argument("--skip-imports", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "verify":
        raise RuntimeError("unsupported command")
    report = verify_runtime(args.project_root, verify_imports=not args.skip_imports)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
