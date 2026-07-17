from __future__ import annotations

import argparse
import hashlib
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
_LOCK_KEYS = {
    "derived_trees",
    "environments",
    "runtime_files",
    "schema_version",
    "sources",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git verification failed for {source}: {result.stderr.strip()}")
    return result.stdout.strip()


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


@dataclass(frozen=True)
class RuntimeArtifactLock:
    path: Path
    environments: Mapping[str, Mapping[str, Any]]
    sources: Mapping[str, Mapping[str, Any]]
    runtime_files: Mapping[str, Mapping[str, Any]]
    derived_trees: Mapping[str, Mapping[str, Any]]

    @classmethod
    def load(cls, project_root: str | Path) -> "RuntimeArtifactLock":
        path = Path(project_root).resolve() / "locks" / "runtime_artifacts.json"
        value = _json(path)
        _fields(value, _LOCK_KEYS, "runtime_artifacts")
        if value["schema_version"] != 1:
            raise ValueError("runtime_artifacts.schema_version must be 1")
        for key in _LOCK_KEYS - {"schema_version"}:
            if not isinstance(value[key], dict):
                raise ValueError(f"runtime_artifacts.{key} must be an object")
        return cls(path, value["environments"], value["sources"], value["runtime_files"], value["derived_trees"])

    def source(self, name: str) -> Mapping[str, Any]:
        try:
            entry = self.sources[name]
        except KeyError as exc:
            raise KeyError(f"runtime artifact lock has no source {name!r}") from exc
        expected = {"commit", "directory", "submodules"}
        if not isinstance(entry, dict) or set(entry) - expected or "commit" not in entry:
            raise ValueError(f"runtime source lock is invalid for {name!r}")
        return entry


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


def _verify_source(name: str, entry: Mapping[str, Any], paths: RuntimePaths) -> Path:
    expected_fields = {"commit", "directory", "submodules"}
    if set(entry) - expected_fields or "commit" not in entry:
        raise ValueError(f"runtime source lock is invalid for {name!r}")
    commit = entry["commit"]
    directory = entry.get("directory", name)
    submodules = entry.get("submodules", {})
    if not isinstance(commit, str) or len(commit) != 40 or not isinstance(directory, str) or not isinstance(submodules, dict):
        raise ValueError(f"runtime source lock is invalid for {name!r}")
    source = paths.source(directory)
    if _git(source, "rev-parse", "HEAD") != commit:
        raise RuntimeError(f"source revision mismatch for {name}")
    dirty = _git(source, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
    if dirty:
        raise RuntimeError(f"source working tree is dirty for {name}: {dirty.splitlines()[0]}")
    for relative, expected in sorted(submodules.items()):
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 40:
            raise ValueError(f"source submodule lock is invalid for {name!r}")
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError(f"source submodule path escapes for {name!r}")
        if _git(path, "rev-parse", "HEAD") != expected:
            raise RuntimeError(f"submodule revision mismatch for {name}:{relative}")
    return source


def _derived_tree_sha256(root: Path, marker_name: str) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.name not in {marker_name, "z"}):
        relative = path.relative_to(root).as_posix().encode()
        if path.is_symlink():
            payload = b"L\0" + os.readlink(path).encode()
        elif path.is_file():
            payload = b"F\0" + bytes.fromhex(_sha256(path))
        elif path.is_dir():
            payload = b"D\0"
        else:
            raise RuntimeError(f"unsupported derived source entry: {path}")
        digest.update(relative + b"\0" + payload + b"\0")
    return digest.hexdigest()


def _verify_runtime_file(name: str, entry: Mapping[str, Any], paths: RuntimePaths) -> Path:
    _fields(entry, {"path_key", "sha256", "size_bytes"}, f"runtime_files.{name}")
    path = paths.artifact(entry["path_key"])
    if not path.is_file() or path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
        raise RuntimeError(f"runtime file identity differs: {name}")
    return path


def _verify_derived_tree(name: str, entry: Mapping[str, Any], paths: RuntimePaths, full_tree: bool) -> Path:
    _fields(
        entry,
        {"marker_path_key", "marker_relative_path", "marker_sha256", "marker_size_bytes", "tree_sha256"},
        f"derived_trees.{name}",
    )
    root = paths.artifact(entry["marker_path_key"])
    marker = (root / entry["marker_relative_path"]).resolve()
    if not marker.is_relative_to(root) or not marker.is_file():
        raise RuntimeError(f"derived tree marker is missing: {name}")
    if marker.stat().st_size != entry["marker_size_bytes"] or _sha256(marker) != entry["marker_sha256"]:
        raise RuntimeError(f"derived tree marker identity differs: {name}")
    payload = _json(marker)
    if payload.get("tree_sha256") != entry["tree_sha256"]:
        raise RuntimeError(f"derived tree marker payload differs: {name}")
    if full_tree and _derived_tree_sha256(root, marker.name) != entry["tree_sha256"]:
        raise RuntimeError(f"derived tree identity differs: {name}")
    return root


def _existing(paths: RuntimePaths) -> None:
    roots = (paths.environment_root, paths.source_root, paths.shared_cache_root)
    missing = [str(path) for path in (*roots, *paths.artifacts.values()) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"declared runtime paths are missing: {missing}")


def verify_runtime(
    project_root: str | Path,
    *,
    verify_imports: bool = True,
    verify_sources: bool = True,
    full_derived_tree: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths = RuntimePaths.load(root)
    lock = RuntimeArtifactLock.load(root)
    assert_clean_import_origin(root)
    _existing(paths)
    environments: list[str] = []
    imports: list[str] = []
    for name, entry in sorted(lock.environments.items()):
        _fields(entry, {"history_sha256", "python_version"}, f"environments.{name}")
        python = paths.environment_python(name)
        history = python.parent.parent / "conda-meta" / "history"
        if not python.is_file() or not history.is_file() or _sha256(history) != entry["history_sha256"]:
            raise RuntimeError(f"environment identity differs: {name}")
        result = subprocess.run(
            [str(python), "-s", "-c", "import platform;print(platform.python_version())"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 or result.stdout.strip() != entry["python_version"]:
            raise RuntimeError(f"environment Python version differs: {name}")
        environments.append(name)
        if verify_imports:
            verify_python_import_origin(python, root, paths)
            imports.append(name)
    sources = []
    if verify_sources:
        for name, entry in sorted(lock.sources.items()):
            _verify_source(name, entry, paths)
            sources.append(name)
    runtime_files = []
    for name, entry in sorted(lock.runtime_files.items()):
        _verify_runtime_file(name, entry, paths)
        runtime_files.append(name)
    derived = []
    derived_markers = []
    for name, entry in sorted(lock.derived_trees.items()):
        _verify_derived_tree(name, entry, paths, full_derived_tree)
        (derived if full_derived_tree else derived_markers).append(name)
    identity_locked_artifacts = {
        entry["path_key"] for entry in lock.runtime_files.values()
    } | {entry["marker_path_key"] for entry in lock.derived_trees.values()}
    declared_only = sorted(set(paths.artifacts) - identity_locked_artifacts)
    return {
        "complete": True,
        "project_root": str(root),
        "runtime_paths": str(paths.config_path),
        "verified": {
            "derived_trees": derived,
            "derived_tree_markers": derived_markers,
            "environment_import_origins": imports,
            "environments": environments,
            "runtime_files": runtime_files,
            "sources": sources,
        },
        "declared_only": {
            "artifacts": declared_only,
            "roots": ["environment_root", "shared_cache_root", "source_root"],
        },
        "verification_scope": {
            "derived_tree_content": full_derived_tree,
            "environment_lock": "Conda history hash and Python version; pip contents are not bit-for-bit locked",
            "path_only_artifacts": "Existence checked; no portable content identity is committed",
            "source_content": "Git commit, clean worktree, and listed submodule commits",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m robot_auto_evolve.runtime_paths")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--project-root", type=Path, default=project_root_from_package())
    verify.add_argument("--skip-imports", action="store_true")
    verify.add_argument("--skip-sources", action="store_true")
    verify.add_argument("--marker-only", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "verify":
        raise RuntimeError("unsupported command")
    report = verify_runtime(
        args.project_root,
        verify_imports=not args.skip_imports,
        verify_sources=not args.skip_sources,
        full_derived_tree=not args.marker_only,
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
