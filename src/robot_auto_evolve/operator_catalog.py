from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

from robot_auto_evolve.config import Profile
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.provenance import BenchmarkPlan
from robot_auto_evolve.service_ports import checked_static_service_port


REQUEST_KIND = "robot_auto_evolve_study_request"
CATALOG_KIND = "robot_auto_evolve_route_catalog"
ROUTE_KIND = "robot_auto_evolve_route"
MODES = frozenset({"full_benchmark", "related_transfer", "seed_transfer"})
EPISODE_SPLIT_KIND = "alternating_scenarios"
EPISODE_SPLIT_FIELDS = {"kind", "evolve_per_task", "held_out_per_task"}
FULL_BENCHMARK_STATUSES = frozenset({"ready", "ready_noncomparable"})
TRANSFER_POLICY = "baseline_vs_frozen_after_finalize"
ADAPTIVE_EVIDENCE_POLICY = "evolve_only"
RUNTIME_PROFILE_KIND = "robot_auto_evolve_runtime_profile"
RUNTIME_CONFIG_KIND = "robot_auto_evolve_runtime_config"
RUNTIME_CONFIG_FIELDS = {
    "schema_version",
    "kind",
    "gpu_ids",
    "render_gpu_ids",
    "workers_per_gpu",
    "workers_per_gpu_with_language",
    "policies_per_gpu",
    "port_offset",
    "vllm",
    "reuse_agent",
    "reuse_sim",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _strict_json(path: Path) -> Any:
    def pairs(items: list[tuple[Any, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if type(key) is not str or key in result:
                raise StrictSchemaError(f"invalid JSON object in {path}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                StrictSchemaError(f"invalid JSON constant {value} in {path}")
            ),
        )
    except StrictSchemaError:
        raise
    except Exception as exc:
        raise StrictSchemaError(f"cannot load JSON {path}: {exc}") from exc


def _exact_fields(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise StrictSchemaError(f"{path}: fields differ")
    return value


def _string(value: Any, path: str) -> str:
    if type(value) is not str or not value:
        raise StrictSchemaError(f"{path}: expected nonempty string")
    return value


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise StrictSchemaError(f"{path}: expected integer >= {minimum}")
    return value


def _relative_path(value: Any, path: str, *, suffix: str | None = None) -> str:
    result = _string(value, path)
    pure = PurePosixPath(result)
    if pure.is_absolute() or ".." in pure.parts or result != pure.as_posix():
        raise StrictSchemaError(f"{path}: expected safe relative path")
    if suffix is not None and pure.suffix != suffix:
        raise StrictSchemaError(f"{path}: expected {suffix} path")
    return result


def _resolve(project_root: Path, relative: str) -> Path:
    root = project_root.resolve()
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise StrictSchemaError("path escapes project root") from exc
    return result


def _sorted_ids(value: Any, path: str, *, nonempty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str or not item for item in value):
        raise StrictSchemaError(f"{path}: expected string list")
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        raise StrictSchemaError(f"{path}: expected sorted unique IDs")
    if nonempty and not result:
        raise StrictSchemaError(f"{path}: expected nonempty IDs")
    return result


def _reference(value: Any, path: str) -> dict[str, str]:
    result = _exact_fields(value, {"path"}, path)
    return {
        "path": _relative_path(result["path"], f"{path}.path", suffix=".json"),
    }


def _selector_matches(episode: Any, selector: Mapping[str, Any]) -> bool:
    if set(selector) not in ({"task_id"}, {"task_id", "scenario_prefix"}):
        raise StrictSchemaError("route task row selector fields differ")
    task_id = _string(selector["task_id"], "route task selector.task_id")
    if episode.task_id != task_id:
        return False
    if "scenario_prefix" in selector:
        prefix = _string(selector["scenario_prefix"], "route task selector.scenario_prefix")
        return episode.scenario_id.startswith(prefix)
    return True


def _filtered_plan(plan: BenchmarkPlan, plan_id: str, units: Sequence[Mapping[str, Any]]) -> BenchmarkPlan:
    episodes = tuple(
        item
        for item in plan.episodes
        if any(_selector_matches(item, unit["row_selector"]) for unit in units)
    )
    if not episodes:
        raise StrictSchemaError("study task selection produced no benchmark rows")
    return BenchmarkPlan(plan_id=plan_id, model_route=plan.model_route, episodes=episodes)


def default_episode_split(
    evolve_per_task: int | None = None,
    held_out_per_task: int | None = None,
) -> dict[str, Any]:
    return {
        "kind": EPISODE_SPLIT_KIND,
        "evolve_per_task": evolve_per_task,
        "held_out_per_task": held_out_per_task,
    }


def _checked_episode_split(value: Any) -> dict[str, Any]:
    split = _exact_fields(value, EPISODE_SPLIT_FIELDS, "study_request.episode_split")
    if split["kind"] != EPISODE_SPLIT_KIND:
        raise StrictSchemaError("study_request.episode_split.kind: unsupported")
    evolve = split["evolve_per_task"]
    held_out = split["held_out_per_task"]
    if (evolve is None) != (held_out is None):
        raise StrictSchemaError(
            "study_request.episode_split: give both evolve_per_task and held_out_per_task, or neither"
        )
    if evolve is not None:
        evolve = _integer(evolve, "study_request.episode_split.evolve_per_task", 1)
        held_out = _integer(held_out, "study_request.episode_split.held_out_per_task", 1)
    return {"kind": EPISODE_SPLIT_KIND, "evolve_per_task": evolve, "held_out_per_task": held_out}


def deal_indices(available: int, evolve_count: int, held_out_count: int) -> tuple[list[int], list[int]]:
    total = evolve_count + held_out_count
    if total > available:
        raise StrictSchemaError(
            f"seed split needs {total} rows per task but only {available} are available"
        )
    evolve: list[int] = []
    held_out: list[int] = []
    for index in range(total):
        if (index * evolve_count) % total < evolve_count:
            evolve.append(index)
        else:
            held_out.append(index)
    return evolve, held_out


def split_plan_by_scenario(
    plan: BenchmarkPlan,
    evolve_plan_id: str,
    held_out_plan_id: str,
    split: Mapping[str, Any],
) -> tuple[BenchmarkPlan, BenchmarkPlan]:
    checked = _checked_episode_split(split)
    by_task: dict[str, list[Any]] = {}
    for item in plan.episodes:
        by_task.setdefault(item.task_id, []).append(item)
    evolve: list[Any] = []
    held_out: list[Any] = []
    for task_id in sorted(by_task):
        rows = sorted(by_task[task_id])
        available = len(rows)
        if checked["evolve_per_task"] is None:
            evolve_count = (available + 1) // 2
            held_out_count = available - evolve_count
        else:
            evolve_count = checked["evolve_per_task"]
            held_out_count = checked["held_out_per_task"]
        if evolve_count < 1 or held_out_count < 1:
            raise StrictSchemaError(
                f"seed split needs at least one row on each side; {task_id} has {available}"
            )
        try:
            evolve_index, held_out_index = deal_indices(available, evolve_count, held_out_count)
        except StrictSchemaError as exc:
            raise StrictSchemaError(f"{exc} ({task_id})") from exc
        evolve.extend(rows[index] for index in evolve_index)
        held_out.extend(rows[index] for index in held_out_index)
    if not evolve or not held_out:
        raise StrictSchemaError("seed split produced an empty half")
    if {item.task_id for item in evolve} != {item.task_id for item in held_out}:
        raise StrictSchemaError("seed split halves cover different tasks")
    if {item.sampling_key() for item in evolve} & {item.sampling_key() for item in held_out}:
        raise StrictSchemaError("seed split halves share an episode")
    return (
        BenchmarkPlan(plan_id=evolve_plan_id, model_route=plan.model_route, episodes=tuple(evolve)),
        BenchmarkPlan(plan_id=held_out_plan_id, model_route=plan.model_route, episodes=tuple(held_out)),
    )


@dataclass(frozen=True)
class StudyRequest:
    mapping: dict[str, Any]
    route_spec: dict[str, Any]
    benchmark_plan: BenchmarkPlan
    evolve_plan: BenchmarkPlan
    transfer_plan: BenchmarkPlan | None
    project_root: Path

    @property
    def study_id(self) -> str:
        return str(self.mapping["study_id"])

    @property
    def route_id(self) -> str:
        return str(self.mapping["route_id"])

    @property
    def mode(self) -> str:
        return str(self.mapping["mode"])

    @property
    def scalar_metric(self) -> str:
        return str(self.mapping["scalar_metric"])

    @property
    def candidate_budget(self) -> int:
        return int(self.mapping["candidate_budget"])

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.mapping))

    @classmethod
    def load(cls, path: str | Path, project_root: str | Path) -> "StudyRequest":
        source = Path(path).resolve()
        root = Path(project_root).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise StrictSchemaError("study request escapes project root") from exc
        return cls.from_mapping(_strict_json(source), root)

    @classmethod
    def from_mapping(cls, value: Any, project_root: str | Path) -> "StudyRequest":
        root = Path(project_root).resolve()
        request = _exact_fields(
            value,
            {
                "schema_version",
                "kind",
                "study_id",
                "route_id",
                "mode",
                "route_spec",
                "profile",
                "profiles",
                "benchmark_plan",
                "standard_source_plan",
                "scalar_metric",
                "task_selection",
                "episode_split",
                "effective_plan",
                "candidate_budget",
                "policies",
            },
            "study_request",
        )
        if request["schema_version"] != 1 or request["kind"] != REQUEST_KIND:
            raise StrictSchemaError("study_request: identity differs")
        study_id = _string(request["study_id"], "study_request.study_id")
        route_id = _string(request["route_id"], "study_request.route_id")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", study_id) is None:
            raise StrictSchemaError("study_request.study_id: invalid")
        mode = _string(request["mode"], "study_request.mode")
        if mode not in MODES:
            raise StrictSchemaError("study_request.mode: unsupported")

        route_reference = _reference(request["route_spec"], "study_request.route_spec")
        route_path = _resolve(root, route_reference["path"])
        if not route_path.is_file() or route_path.is_symlink():
            raise StrictSchemaError("study_request.route_spec: file differs")
        route_spec = _strict_json(route_path)
        if route_spec.get("schema_version") != 1 or route_spec.get("kind") != ROUTE_KIND:
            raise StrictSchemaError("route specification identity differs")
        if route_spec.get("route_id") != route_id or route_spec.get("integration_ready") is not True:
            raise StrictSchemaError("route specification route identity differs")
        benchmark = route_spec.get("benchmark")
        if not isinstance(benchmark, dict) or benchmark.get("status") not in FULL_BENCHMARK_STATUSES:
            raise StrictSchemaError("route full benchmark is not launch-ready")

        profile_reference = _reference(request["profile"], "study_request.profile")
        expected_profile = route_spec.get("profile")
        if not isinstance(expected_profile, dict) or profile_reference != {
            "path": expected_profile.get("path"),
        }:
            raise StrictSchemaError("study_request.profile: route profile differs")
        profile_path = _resolve(root, profile_reference["path"])
        if not profile_path.is_file() or profile_path.is_symlink():
            raise StrictSchemaError("study_request.profile: file differs")
        profile_values = request["profiles"]
        expected_profiles = route_spec.get("profiles")
        if not isinstance(profile_values, dict) or not isinstance(expected_profiles, dict):
            raise StrictSchemaError("study_request.profiles: profile set differs")
        if set(profile_values) != set(expected_profiles) or not profile_values:
            raise StrictSchemaError("study_request.profiles: profile keys differ")
        checked_profiles = {}
        for key in sorted(profile_values):
            _string(key, "study_request.profiles key")
            if re.fullmatch(r"[a-z0-9][a-z0-9_]{0,127}", key) is None:
                raise StrictSchemaError("study_request.profiles: invalid profile key")
            reference = _reference(profile_values[key], f"study_request.profiles.{key}")
            expected = expected_profiles[key]
            if reference != {"path": expected.get("path")}:
                raise StrictSchemaError(f"study_request.profiles.{key}: route profile differs")
            source = _resolve(root, reference["path"])
            if not source.is_file() or source.is_symlink():
                raise StrictSchemaError(f"study_request.profiles.{key}: file differs")
            checked_profiles[key] = reference
        primary_key = route_spec.get("primary_profile_key")
        if type(primary_key) is not str or primary_key not in checked_profiles or checked_profiles[primary_key] != profile_reference:
            raise StrictSchemaError("study_request.profile: primary profile differs")

        plan_reference = _reference(request["benchmark_plan"], "study_request.benchmark_plan")
        if plan_reference != benchmark.get("plan"):
            raise StrictSchemaError("study_request.benchmark_plan: route plan differs")
        benchmark_plan = BenchmarkPlan.load(_resolve(root, plan_reference["path"]))

        source_reference = _reference(request["standard_source_plan"], "study_request.standard_source_plan")
        if source_reference != benchmark.get("standard_source_plan"):
            raise StrictSchemaError("study_request.standard_source_plan: route source differs")
        source_plan = BenchmarkPlan.load(_resolve(root, source_reference["path"]))
        source_rows = {_canonical_bytes(item.to_mapping()) for item in source_plan.episodes}
        if any(_canonical_bytes(item.to_mapping()) not in source_rows for item in benchmark_plan.episodes):
            raise StrictSchemaError("route benchmark row differs from the exact standard source")

        scalar_metric = _string(request["scalar_metric"], "study_request.scalar_metric")
        if scalar_metric != benchmark.get("metric"):
            raise StrictSchemaError("study_request.scalar_metric: route metric differs")
        selection = _exact_fields(
            request["task_selection"],
            {"evolve_task_ids", "transfer_task_ids"},
            "study_request.task_selection",
        )
        evolve_ids = _sorted_ids(selection["evolve_task_ids"], "study_request.task_selection.evolve_task_ids", nonempty=True)
        transfer_ids = _sorted_ids(
            selection["transfer_task_ids"], "study_request.task_selection.transfer_task_ids", nonempty=False
        )
        if set(evolve_ids) & set(transfer_ids):
            raise StrictSchemaError("study task selections overlap")
        units = benchmark.get("task_units")
        if not isinstance(units, list):
            raise StrictSchemaError("route benchmark task units differ")
        units_by_id: dict[str, Mapping[str, Any]] = {}
        for unit in units:
            if not isinstance(unit, dict) or type(unit.get("id")) is not str or unit["id"] in units_by_id:
                raise StrictSchemaError("route benchmark task units differ")
            units_by_id[unit["id"]] = unit
        if set(units_by_id) != set(benchmark.get("task_ids", [])):
            raise StrictSchemaError("route benchmark task IDs differ")
        if not set((*evolve_ids, *transfer_ids)) <= set(units_by_id):
            raise StrictSchemaError("study request contains unknown route tasks")
        if mode == "full_benchmark" and (set(evolve_ids) != set(units_by_id) or transfer_ids):
            raise StrictSchemaError("full_benchmark mode requires every route task and no transfer tasks")
        if mode == "related_transfer" and not transfer_ids:
            raise StrictSchemaError("related_transfer mode requires nonempty evolve and transfer task sets")
        if mode == "seed_transfer" and (set(evolve_ids) != set(units_by_id) or transfer_ids):
            raise StrictSchemaError("seed_transfer mode requires every route task and no transfer tasks")
        episode_split = request["episode_split"]
        if mode == "seed_transfer":
            episode_split = _checked_episode_split(episode_split)
        elif episode_split is not None:
            raise StrictSchemaError("study_request.episode_split: only seed_transfer mode carries a split")
        evolve_units = [units_by_id[item] for item in evolve_ids]
        transfer_units = [units_by_id[item] for item in transfer_ids]
        evolve_row_tasks = {item["row_selector"]["task_id"] for item in evolve_units}
        transfer_row_tasks = {item["row_selector"]["task_id"] for item in transfer_units}
        if evolve_row_tasks & transfer_row_tasks:
            raise StrictSchemaError("evolve and transfer selections overlap in underlying benchmark task_id")
        if mode == "seed_transfer":
            evolve_plan, transfer_plan = split_plan_by_scenario(
                _filtered_plan(benchmark_plan, f"{study_id}_all", evolve_units),
                f"{study_id}_evolve",
                f"{study_id}_transfer",
                episode_split,
            )
        else:
            evolve_plan = _filtered_plan(benchmark_plan, f"{study_id}_evolve", evolve_units)
            transfer_plan = (
                None
                if not transfer_units
                else _filtered_plan(benchmark_plan, f"{study_id}_transfer", transfer_units)
            )

        effective = _exact_fields(
            request["effective_plan"],
            {"evolve_episode_count", "transfer_episode_count"},
            "study_request.effective_plan",
        )
        expected_effective = {
            "evolve_episode_count": len(evolve_plan.episodes),
            "transfer_episode_count": 0 if transfer_plan is None else len(transfer_plan.episodes),
        }
        if effective != expected_effective:
            raise StrictSchemaError("study_request.effective_plan: filtered standard rows differ")
        candidate_budget = _integer(request["candidate_budget"], "study_request.candidate_budget", 1)
        if candidate_budget != route_spec.get("defaults", {}).get("candidate_budget"):
            raise StrictSchemaError("study_request.candidate_budget: route budget differs")

        policies = _exact_fields(
            request["policies"],
            {"adaptive_evidence", "transfer_evaluation"},
            "study_request.policies",
        )
        if policies != {
            "adaptive_evidence": ADAPTIVE_EVIDENCE_POLICY,
            "transfer_evaluation": TRANSFER_POLICY,
        }:
            raise StrictSchemaError("study_request.policies: leakage policy differs")
        stable = json.loads(json.dumps(request))
        return cls(stable, route_spec, benchmark_plan, evolve_plan, transfer_plan, root)


def load_catalog(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = root / "routes" / "catalog.json"
    catalog = _strict_json(path)
    if catalog.get("schema_version") != 1 or catalog.get("kind") != CATALOG_KIND:
        raise StrictSchemaError("route catalog identity differs")
    routes = catalog.get("routes")
    if not isinstance(routes, list) or not all(isinstance(item, dict) for item in routes):
        raise StrictSchemaError("route catalog routes differ")
    route_ids = [item.get("route_id") for item in routes]
    if route_ids != sorted(set(route_ids)) or not all(isinstance(x, str) and x for x in route_ids):
        raise StrictSchemaError("route catalog route ids differ")
    canonical = sorted(item["route_id"] for item in routes if item.get("canonical_full_benchmark") is True)
    catalog["integration_ready_route_ids"] = list(route_ids)
    catalog["full_benchmark_ready_route_ids"] = sorted(
        item["route_id"] for item in routes if item.get("full_benchmark_status") in FULL_BENCHMARK_STATUSES
    )
    catalog["canonical_full_benchmark_route_ids"] = canonical
    catalog["slice_and_standalone_route_ids"] = sorted(set(route_ids) - set(canonical))
    return catalog


def load_route_spec(project_root: str | Path, route_id: str) -> tuple[dict[str, Any], str]:
    root = Path(project_root).resolve()
    catalog = load_catalog(root)
    rows = [item for item in catalog["routes"] if item["route_id"] == route_id]
    if len(rows) != 1:
        blocked = [item for item in catalog.get("blocked_routes", []) if item.get("route_id") == route_id]
        if blocked:
            raise PermissionError(blocked[0]["blocker"])
        raise StrictSchemaError(f"unknown route: {route_id}")
    reference = _reference(rows[0]["spec"], "route_catalog.spec")
    path = _resolve(root, reference["path"])
    if not path.is_file() or path.is_symlink():
        raise StrictSchemaError("route specification file differs")
    spec = _strict_json(path)
    if spec.get("route_id") != route_id or spec.get("kind") != ROUTE_KIND:
        raise StrictSchemaError("route specification identity differs")
    return spec, reference["path"]


def list_route_tasks(project_root: str | Path, route_id: str) -> tuple[dict[str, Any], ...]:
    spec, _ = load_route_spec(project_root, route_id)
    tasks = spec.get("benchmark", {}).get("task_units")
    if not isinstance(tasks, list):
        raise StrictSchemaError("route task catalog differs")
    return tuple(json.loads(json.dumps(item)) for item in tasks)


def build_runtime_config(
    project_root: str | Path,
    route_id: str,
    *,
    gpu_ids: Sequence[int] = (0, 1),
    render_gpu_ids: Sequence[int] | None = None,
    workers_per_gpu: int | None = None,
    workers_per_gpu_with_language: int | None = None,
    policies_per_gpu: int | None = None,
    port_offset: int = 0,
    vllm: bool = True,
    reuse_agent: bool = True,
    reuse_sim: bool = True,
) -> "RuntimeConfig":
    spec, _ = load_route_spec(project_root, route_id)
    defaults = spec["defaults"]
    with_language = (
        defaults["workers_per_gpu"]
        if workers_per_gpu_with_language is None
        else workers_per_gpu_with_language
    )
    without_language = (
        defaults.get("workers_per_gpu_no_language", defaults["workers_per_gpu"])
        if workers_per_gpu is None
        else workers_per_gpu
    )
    return RuntimeConfig.from_mapping(
        {
            "schema_version": 1,
            "kind": RUNTIME_CONFIG_KIND,
            "gpu_ids": list(gpu_ids),
            "render_gpu_ids": None if render_gpu_ids is None else list(render_gpu_ids),
            "workers_per_gpu": without_language,
            "workers_per_gpu_with_language": with_language,
            "policies_per_gpu": (
                defaults.get("policies_per_gpu", 1) if policies_per_gpu is None else policies_per_gpu
            ),
            "port_offset": port_offset,
            "vllm": bool(vllm),
            "reuse_agent": bool(reuse_agent),
            "reuse_sim": bool(reuse_sim),
        }
    )


def build_study_request(
    project_root: str | Path,
    route_id: str,
    run_id: str,
    *,
    evolve_task_ids: Sequence[str] = (),
    transfer_task_ids: Sequence[str] = (),
    task_preset: str | None = None,
    seed_split: bool = False,
    seed_split_evolve: int | None = None,
    seed_split_held_out: int | None = None,
) -> StudyRequest:
    root = Path(project_root).resolve()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise StrictSchemaError("run ID is invalid")
    spec, spec_path = load_route_spec(root, route_id)
    benchmark = spec["benchmark"]
    if benchmark["status"] not in FULL_BENCHMARK_STATUSES:
        raise PermissionError(benchmark["blocker"])
    available = tuple(benchmark["task_ids"])
    explicit_evolve = tuple(evolve_task_ids)
    explicit_transfer = tuple(transfer_task_ids)
    if task_preset is not None:
        if explicit_evolve or explicit_transfer:
            raise StrictSchemaError("--task-preset cannot be combined with explicit task flags")
        matches = [
            item
            for item in spec.get("related_transfer_presets", [])
            if item.get("preset_id") == task_preset
        ]
        if len(matches) != 1:
            raise StrictSchemaError(f"task preset is not valid for this route: {task_preset}")
        explicit_evolve = tuple(matches[0]["evolve_task_ids"])
        explicit_transfer = tuple(matches[0]["transfer_task_ids"])
    if bool(explicit_evolve) != bool(explicit_transfer):
        raise StrictSchemaError("explicit task selection requires both --evolve-task and --transfer-task")
    if len(set(explicit_evolve)) != len(explicit_evolve) or len(set(explicit_transfer)) != len(explicit_transfer):
        raise StrictSchemaError("task flags must not repeat an ID")
    evolve = tuple(sorted(explicit_evolve or available))
    transfer = tuple(sorted(explicit_transfer))
    if not set((*evolve, *transfer)) <= set(available):
        raise StrictSchemaError("task flag is not valid for this route")
    if set(evolve) & set(transfer):
        raise StrictSchemaError("evolve and transfer task flags must be disjoint")
    if seed_split:
        if explicit_evolve or explicit_transfer:
            raise StrictSchemaError("--seed-split runs the whole route task set and takes no task flags")
        mode = "seed_transfer"
    else:
        mode = "full_benchmark" if not explicit_evolve and not explicit_transfer else "related_transfer"
    if not seed_split and (seed_split_evolve is not None or seed_split_held_out is not None):
        raise StrictSchemaError("per-task episode counts only apply with --seed-split")
    episode_split = default_episode_split(seed_split_evolve, seed_split_held_out) if seed_split else None
    plan_reference = benchmark["plan"]
    if not isinstance(plan_reference, dict):
        raise PermissionError(benchmark["blocker"])
    benchmark_plan = BenchmarkPlan.load(_resolve(root, plan_reference["path"]))
    units_by_id = {item["id"]: item for item in benchmark["task_units"]}
    if {
        units_by_id[item]["row_selector"]["task_id"] for item in evolve
    } & {
        units_by_id[item]["row_selector"]["task_id"] for item in transfer
    }:
        raise StrictSchemaError("evolve and transfer selections overlap in underlying benchmark task_id")
    if seed_split:
        evolve_plan, transfer_plan = split_plan_by_scenario(
            _filtered_plan(
                benchmark_plan,
                f"{route_id}_{run_id}_all",
                [units_by_id[item] for item in evolve],
            ),
            f"{route_id}_{run_id}_evolve",
            f"{route_id}_{run_id}_transfer",
            episode_split,
        )
    else:
        evolve_plan = _filtered_plan(
            benchmark_plan,
            f"{route_id}_{run_id}_evolve",
            [units_by_id[item] for item in evolve],
        )
        transfer_plan = (
            None
            if not transfer
            else _filtered_plan(
                benchmark_plan,
                f"{route_id}_{run_id}_transfer",
                [units_by_id[item] for item in transfer],
            )
        )
    mapping = {
        "schema_version": 1,
        "kind": REQUEST_KIND,
        "study_id": f"{route_id}_{run_id}",
        "route_id": route_id,
        "mode": mode,
        "route_spec": {"path": spec_path},
        "profile": {
            "path": spec["profile"]["path"],
        },
        "profiles": {
            key: {"path": value["path"]}
            for key, value in sorted(spec["profiles"].items())
        },
        "benchmark_plan": plan_reference,
        "standard_source_plan": benchmark["standard_source_plan"],
        "scalar_metric": benchmark["metric"],
        "task_selection": {
            "evolve_task_ids": list(evolve),
            "transfer_task_ids": list(transfer),
        },
        "episode_split": episode_split,
        "effective_plan": {
            "evolve_episode_count": len(evolve_plan.episodes),
            "transfer_episode_count": 0 if transfer_plan is None else len(transfer_plan.episodes),
        },
        "candidate_budget": spec["defaults"]["candidate_budget"],
        "policies": {
            "adaptive_evidence": ADAPTIVE_EVIDENCE_POLICY,
            "transfer_evaluation": TRANSFER_POLICY,
        },
    }
    return StudyRequest.from_mapping(mapping, root)


@dataclass(frozen=True)
class RuntimeConfig:
    mapping: dict[str, Any]

    @property
    def gpu_ids(self) -> tuple[int, ...]:
        return tuple(self.mapping["gpu_ids"])

    @property
    def render_gpu_ids(self) -> tuple[int, ...] | None:
        value = self.mapping["render_gpu_ids"]
        return None if value is None else tuple(value)

    @property
    def workers_per_gpu(self) -> int:
        return int(self.mapping["workers_per_gpu"])

    @property
    def workers_per_gpu_with_language(self) -> int:
        return int(self.mapping["workers_per_gpu_with_language"])

    @property
    def policies_per_gpu(self) -> int:
        return int(self.mapping["policies_per_gpu"])

    @property
    def port_offset(self) -> int:
        return int(self.mapping["port_offset"])

    @property
    def vllm(self) -> bool:
        return bool(self.mapping["vllm"])

    @property
    def reuse_agent(self) -> bool:
        return bool(self.mapping["reuse_agent"])

    @property
    def reuse_sim(self) -> bool:
        return bool(self.mapping["reuse_sim"])

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.mapping))

    @classmethod
    def from_mapping(cls, value: Any) -> "RuntimeConfig":
        config = _exact_fields(value, RUNTIME_CONFIG_FIELDS, "runtime_config")
        if config["schema_version"] != 1 or config["kind"] != RUNTIME_CONFIG_KIND:
            raise StrictSchemaError("runtime_config: identity differs")
        if any(type(config[name]) is not bool for name in ("vllm", "reuse_agent", "reuse_sim")):
            raise StrictSchemaError("runtime_config.vllm/reuse_agent/reuse_sim: expected bool")
        gpu_ids = config["gpu_ids"]
        if (
            not isinstance(gpu_ids, list)
            or len(gpu_ids) < 2
            or any(type(item) is not int or item < 0 for item in gpu_ids)
            or gpu_ids != sorted(set(gpu_ids))
        ):
            raise StrictSchemaError("runtime_config.gpu_ids: expected at least two sorted unique IDs")
        render_ids = config["render_gpu_ids"]
        if render_ids is not None and (
            not isinstance(render_ids, list)
            or len(render_ids) != len(gpu_ids)
            or any(type(item) is not int or item not in gpu_ids for item in render_ids)
        ):
            raise StrictSchemaError("runtime_config.render_gpu_ids: expected one pool GPU per pool GPU")
        _integer(config["workers_per_gpu"], "runtime_config.workers_per_gpu", 1)
        _integer(config["workers_per_gpu_with_language"], "runtime_config.workers_per_gpu_with_language", 1)
        _integer(config["policies_per_gpu"], "runtime_config.policies_per_gpu", 1)
        _integer(config["port_offset"], "runtime_config.port_offset", 0)
        return cls(json.loads(json.dumps(config)))

    @classmethod
    def load(cls, run_root: str | Path) -> "RuntimeConfig":
        return cls.from_mapping(_strict_json(Path(run_root).resolve() / "runtime_config.json"))


def materialize_runtime_config(config: RuntimeConfig, run_root: str | Path) -> Path:
    target_root = Path(run_root).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / "runtime_config.json"
    _overwrite(target, _pretty_bytes(config.to_mapping()))
    return target


def _overwrite(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def materialize_study_request(
    request: StudyRequest,
    run_root: str | Path,
) -> Path:
    target_root = Path(run_root).resolve()
    project_root = request.project_root.resolve()
    expected_parent = project_root / "runs"
    try:
        relative = target_root.relative_to(expected_parent)
    except ValueError as exc:
        raise StrictSchemaError("run root must stay directly under runs") from exc
    if len(relative.parts) != 1 or target_root.name != request.study_id:
        raise StrictSchemaError("run root must match the study ID")
    target_root.mkdir(parents=True, exist_ok=True)
    if target_root.is_symlink():
        raise StrictSchemaError("run root must not be a symlink")
    target = target_root / "study_request.json"
    payload = _pretty_bytes(request.to_mapping())
    if target.exists():
        if not target.is_file() or target.is_symlink() or target.read_bytes() != payload:
            raise RuntimeError("existing study request differs")
        StudyRequest.load(target, project_root)
        return target
    _overwrite(target, payload)
    StudyRequest.load(target, project_root)
    return target


def _offset_endpoint(endpoint: str, port_offset: int, ordinal: int = 0) -> str:
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise StrictSchemaError(f"unsupported source service endpoint: {endpoint}")
    port = checked_static_service_port(parsed.port + port_offset + ordinal)
    return urlunparse(("http", f"127.0.0.1:{port}", "", "", "", ""))


def _derived_runtime_profile(
    request: StudyRequest,
    runtime: RuntimeConfig,
    profile_key: str,
    source_reference: Mapping[str, str],
) -> tuple[Profile, Profile, bytes, list[dict[str, Any]]]:
    root = request.project_root
    source_path = _resolve(root, source_reference["path"])
    source = Profile.load(source_path, project_root=root)
    mapping = deepcopy(source.to_mapping())
    gpu_ids = runtime.gpu_ids
    port_offset = runtime.port_offset
    use_vllm = runtime.vllm
    source_replicas = mapping["policy"]["replicas"]
    if (
        mapping["policy"]["deployment_mode"] != "replicated"
        or len(source_replicas) != 2
        or [item["identity"]["gpu_ids"] for item in source_replicas] != [[0], [1]]
        or [item["identity"]["replica_id"] for item in source_replicas] != ["gpu0", "gpu1"]
    ):
        raise StrictSchemaError("source profile must use the checked two-replica logical GPU layout")
    if _offset_endpoint(source_replicas[0]["endpoint"], 0, 1) != source_replicas[1]["endpoint"]:
        raise StrictSchemaError("source policy endpoints must be consecutive")
    source_template = source_replicas[0]
    derived_replicas = []
    service_records = []
    slots = [
        (gpu_id, copy_index)
        for gpu_id in gpu_ids
        for copy_index in range(runtime.policies_per_gpu)
    ]
    for ordinal, (gpu_id, copy_index) in enumerate(slots):
        replica = deepcopy(source_template)
        replica["endpoint"] = _offset_endpoint(source_template["endpoint"], port_offset, ordinal)
        replica["identity"]["gpu_ids"] = [gpu_id]
        replica["identity"]["replica_id"] = (
            f"gpu{gpu_id}" if runtime.policies_per_gpu == 1 else f"gpu{gpu_id}c{copy_index}"
        )
        derived_replicas.append(replica)
        service_records.append(
            {
                "kind": "policy",
                "service_name": replica["identity"]["service_name"],
                "source_endpoint": source_template["endpoint"],
                "runtime_endpoint": replica["endpoint"],
                "source_gpu_ids": source_template["identity"]["gpu_ids"],
                "runtime_gpu_ids": replica["identity"]["gpu_ids"],
                "source_replica_id": source_template["identity"]["replica_id"],
                "runtime_replica_id": replica["identity"]["replica_id"],
            }
        )
    mapping["policy"]["replicas"] = derived_replicas
    logical_gpu_map = {0: gpu_ids[0], 1: gpu_ids[1]}
    for tool in mapping["tools"]:
        service = tool["service"]
        if service is None:
            continue
        source_service = deepcopy(service)
        source_gpu_ids = source_service["identity"]["gpu_ids"]
        if len(source_gpu_ids) != 1 or source_gpu_ids[0] not in logical_gpu_map:
            raise StrictSchemaError("source tool must use logical GPU 0 or 1")
        physical_gpu = logical_gpu_map[source_gpu_ids[0]]
        replica_id = source_service["identity"]["replica_id"]
        match = re.fullmatch(r"gpu[01](.*)", replica_id)
        if match is None:
            raise StrictSchemaError("source tool replica ID must begin with gpu0 or gpu1")
        service["endpoint"] = _offset_endpoint(source_service["endpoint"], port_offset)
        service["identity"]["gpu_ids"] = [physical_gpu]
        service["identity"]["replica_id"] = f"gpu{physical_gpu}{match.group(1)}"
        _vllm_swap = {
            "qwen-language": ("openai-compatible-language", "openai_language"),
            "molmo2-vision": ("openai-compatible-vision", "openai_vision"),
            "qwen-vision": ("openai-compatible-vision", "openai_vision"),
        }
        if use_vllm and source_service["identity"]["service_name"] in _vllm_swap:
            new_service_name, _ = _vllm_swap[source_service["identity"]["service_name"]]
            service["identity"]["service_name"] = new_service_name
        service_records.append(
            {
                "kind": "tool",
                "service_name": service["identity"]["service_name"],
                "source_endpoint": source_service["endpoint"],
                "runtime_endpoint": service["endpoint"],
                "source_gpu_ids": source_gpu_ids,
                "runtime_gpu_ids": service["identity"]["gpu_ids"],
                "source_replica_id": replica_id,
                "runtime_replica_id": service["identity"]["replica_id"],
            }
        )
    mapping["resources"] = {
        "mode": "two_gpu" if len(gpu_ids) == 2 else "multi_gpu",
        "gpu_ids": list(gpu_ids),
    }
    endpoints = [item["endpoint"] for item in derived_replicas] + [
        tool["service"]["endpoint"] for tool in mapping["tools"] if tool["service"] is not None
    ]
    if len(set(endpoints)) != len(endpoints):
        raise StrictSchemaError(
            "runtime profile: policies_per_gpu is too large, the policy ports reach the tool ports"
        )
    derived = Profile.from_mapping(mapping)
    derived.validate(derived.episode_plan.load(root))
    profile_payload = _pretty_bytes(derived.to_mapping())
    return source, derived, profile_payload, service_records


def _runtime_profile_payloads(
    request: StudyRequest,
    runtime: RuntimeConfig,
) -> tuple[dict[str, bytes], str, bytes]:
    profile_payloads = {}
    profile_records = {}
    derived_profiles = {}
    service_records = None
    request_profiles = request.mapping["profiles"]
    primary_key = request.route_spec["primary_profile_key"]
    for key in sorted(request_profiles):
        source_reference = request_profiles[key]
        _, derived, payload, records = _derived_runtime_profile(request, runtime, key, source_reference)
        profile_payloads[key] = payload
        derived_profiles[key] = derived
        profile_records[key] = {
            "source_profile": source_reference["path"],
            "runtime_profile": f"profiles/{key}.json",
        }
        if service_records is None:
            service_records = records
        elif records != service_records:
            raise StrictSchemaError("derived runtime profile services differ across profile set")
    if primary_key not in profile_payloads:
        raise StrictSchemaError("primary runtime profile key differs")
    primary = derived_profiles[primary_key]
    for key, derived in derived_profiles.items():
        if (
            derived.policy.to_mapping() != primary.policy.to_mapping()
            or [item.to_mapping() for item in derived.tools] != [item.to_mapping() for item in primary.tools]
            or derived.resources != primary.resources
        ):
            raise StrictSchemaError(f"derived runtime profile shared services differ: {key}")
    provenance = {
        "schema_version": 1,
        "kind": RUNTIME_PROFILE_KIND,
        "study_id": request.study_id,
        "primary_profile_key": primary_key,
        "profiles": profile_records,
        "runtime_profile": "profile.json",
        "runtime_config": runtime.to_mapping(),
        "services": service_records,
    }
    return profile_payloads, primary_key, _pretty_bytes(provenance)


def materialize_runtime_profile(
    request: StudyRequest,
    runtime: RuntimeConfig,
    run_root: str | Path,
) -> Path:
    root = Path(run_root).resolve()
    expected = request.project_root / "runs" / request.study_id
    if root != expected:
        raise StrictSchemaError("runtime profile run root differs from study ID")
    runtime_root = root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    if runtime_root.is_symlink():
        raise StrictSchemaError("runtime profile directory must not be a symlink")
    profiles_root = runtime_root / "profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)
    if profiles_root.is_symlink():
        raise StrictSchemaError("runtime profiles directory must not be a symlink")
    profile_payloads, primary_key, provenance_payload = _runtime_profile_payloads(request, runtime)
    profile_path = runtime_root / "profile.json"
    provenance_path = runtime_root / "profile_materialization.json"
    for path, payload in [
        (profile_path, profile_payloads[primary_key]),
        (provenance_path, provenance_payload),
        *((profiles_root / f"{key}.json", payload) for key, payload in sorted(profile_payloads.items())),
    ]:
        _overwrite(path, payload)
    Profile.load(profile_path, project_root=request.project_root)
    return profile_path


def materialize_runtime_profiles(
    request: StudyRequest,
    runtime: RuntimeConfig,
    run_root: str | Path,
) -> dict[str, Path]:
    materialize_runtime_profile(request, runtime, run_root)
    root = Path(run_root).resolve() / "runtime" / "profiles"
    result = {key: root / f"{key}.json" for key in sorted(request.mapping["profiles"])}
    for key, path in result.items():
        profile = Profile.load(path, project_root=request.project_root)
        if profile.environment.suite != key:
            raise StrictSchemaError(f"runtime profile suite differs: {key}")
    return result


__all__ = [
    "ADAPTIVE_EVIDENCE_POLICY",
    "FULL_BENCHMARK_STATUSES",
    "REQUEST_KIND",
    "RUNTIME_CONFIG_KIND",
    "RuntimeConfig",
    "StudyRequest",
    "TRANSFER_POLICY",
    "build_runtime_config",
    "build_study_request",
    "list_route_tasks",
    "load_catalog",
    "load_route_spec",
    "materialize_runtime_config",
    "materialize_study_request",
    "materialize_runtime_profile",
    "materialize_runtime_profiles",
]
