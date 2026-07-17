from __future__ import annotations

import re
from pathlib import Path
from types import MappingProxyType

from robot_auto_evolve.protocol import StrictSchemaError

from .libero_suites import LIBERO_SUITE_TASKS


BASE_SUITES = ("spatial", "object", "goal", "10")
HARNESS_PERTURBATIONS = ("task", "swap")
PUBLIC_PERTURBATIONS = ("lan", "object", "swap", "task")
HORIZONS = MappingProxyType({"spatial": 220, "object": 280, "goal": 300, "10": 520})


def profile_suite(base_suite: str, perturbation: str) -> str:
    if base_suite not in BASE_SUITES or perturbation not in PUBLIC_PERTURBATIONS:
        raise StrictSchemaError("LIBERO-Pro suite is unsupported")
    return f"libero_pro_{base_suite}_{perturbation}"


def upstream_suite(value: str) -> str:
    match = re.fullmatch(r"libero_pro_(spatial|object|goal|10)_(lan|object|swap|task)", value)
    if match is None:
        raise StrictSchemaError("LIBERO-Pro profile suite is unsupported")
    return f"libero_{match.group(1)}_{match.group(2)}"


def split_suite(value: str) -> tuple[str, str]:
    upstream = upstream_suite(value)
    _, base_suite, perturbation = upstream.split("_", 2)
    return base_suite, perturbation


def base_tasks(base_suite: str) -> tuple[str, ...]:
    if base_suite not in BASE_SUITES:
        raise StrictSchemaError("LIBERO-Pro base suite is unsupported")
    return tuple(LIBERO_SUITE_TASKS[f"libero_{base_suite}"])


def task_id(suite: str, task_slug: str) -> str:
    base_suite, _ = split_suite(suite)
    if task_slug not in base_tasks(base_suite):
        raise StrictSchemaError("LIBERO-Pro task is absent from its base suite")
    return f"{suite}::{task_slug}"


def split_task_id(value: str) -> tuple[str, str]:
    suite, separator, task_slug = value.partition("::")
    if not separator:
        raise StrictSchemaError("LIBERO-Pro task id must be namespaced")
    if task_id(suite, task_slug) != value:
        raise StrictSchemaError("LIBERO-Pro task id is invalid")
    return suite, task_slug


HARNESS_SUITES = tuple(
    profile_suite(base_suite, perturbation)
    for base_suite in BASE_SUITES
    for perturbation in HARNESS_PERTURBATIONS
)
PUBLIC_SUITES = tuple(
    profile_suite(base_suite, perturbation)
    for base_suite in BASE_SUITES
    for perturbation in PUBLIC_PERTURBATIONS
)
HARNESS_TASK_SUITE = MappingProxyType(
    {
        task_id(suite, task_slug): suite
        for suite in HARNESS_SUITES
        for task_slug in base_tasks(split_suite(suite)[0])
    }
)


def harness_protocol(suite: str) -> str:
    if suite not in HARNESS_SUITES:
        raise StrictSchemaError("suite is outside the Harness-VLA paper-v3 protocol")
    return f"rlinf_pi05_{suite}_paper_v3_10_seed_v1"


HARNESS_PROTOCOLS = MappingProxyType(
    {suite: MappingProxyType({harness_protocol(suite): HORIZONS[split_suite(suite)[0]]}) for suite in HARNESS_SUITES}
)


def one_step_smoke_protocol(suite: str) -> str:
    if suite not in HARNESS_SUITES:
        raise StrictSchemaError("suite is outside the RLinf pi0.5 LIBERO-Pro smoke protocol")
    return f"rlinf_pi05_{suite}_one_step_smoke_v1"


ONE_STEP_SMOKE_PROTOCOLS = MappingProxyType(
    {suite: MappingProxyType({one_step_smoke_protocol(suite): 1}) for suite in HARNESS_SUITES}
)


_LANGUAGE = re.compile(r"^\s*\(:language\s+(.+?)\s*\)\s*$", re.MULTILINE)
_GOAL = re.compile(r"\(:goal\b", re.IGNORECASE)


def parse_bddl_language(path: str | Path) -> str:
    source = Path(path)
    value = source.read_text(encoding="utf-8")
    matches = _LANGUAGE.findall(value)
    if len(matches) != 1:
        raise StrictSchemaError(f"LIBERO-Pro BDDL must contain one language field: {source}")
    language = " ".join(matches[0].split())
    if not language:
        raise StrictSchemaError("LIBERO-Pro BDDL language is empty")
    return language


def parse_bddl_goal(path: str | Path) -> str:
    source = Path(path)
    value = source.read_text(encoding="utf-8")
    matches = tuple(_GOAL.finditer(value))
    if len(matches) != 1:
        raise StrictSchemaError(f"LIBERO-Pro BDDL must contain one goal field: {source}")
    start = matches[0].start()
    depth = 0
    end = None
    for index, character in enumerate(value[start:], start=start):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
        if depth < 0:
            break
    if end is None or depth != 0:
        raise StrictSchemaError(f"LIBERO-Pro BDDL goal is unbalanced: {source}")
    tokens = re.findall(r"\(|\)|[^\s()]+", value[start:end])
    if not tokens:
        raise StrictSchemaError("LIBERO-Pro BDDL goal is empty")
    return " ".join(tokens)


def _task_split(base_suite: str, evolve: tuple[int, ...], transfer: tuple[int, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tasks = base_tasks(base_suite)
    if sorted((*evolve, *transfer)) != list(range(len(tasks))):
        raise RuntimeError("LIBERO-Pro transfer split must partition its base suite")
    return tuple(tasks[index] for index in evolve), tuple(tasks[index] for index in transfer)


_SWAP_SPLITS = MappingProxyType(
    {
        "spatial": _task_split("spatial", (0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),
        "object": _task_split("object", (0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),
        "goal": _task_split("goal", (0, 1, 2, 3, 7), (4, 5, 6, 8, 9)),
        "10": _task_split("10", (0, 2, 3, 4, 5), (1, 6, 7, 8, 9)),
    }
)
_TASK_SPLITS = MappingProxyType(
    {
        "spatial": _task_split("spatial", (3, 4, 6, 7, 9), (0, 1, 2, 5, 8)),
        "object": _task_split("object", (0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),
        "goal": _task_split("goal", (0, 2, 5, 6, 7), (1, 3, 4, 8, 9)),
        "10": _task_split("10", (0, 2, 3, 4, 5), (1, 6, 7, 8, 9)),
    }
)
RELATED_TRANSFER_TASKS = MappingProxyType(
    {
        **{(base_suite, "swap"): value for base_suite, value in _SWAP_SPLITS.items()},
        **{(base_suite, "task"): value for base_suite, value in _TASK_SPLITS.items()},
    }
)
