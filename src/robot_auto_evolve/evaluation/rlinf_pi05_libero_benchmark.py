from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from robot_auto_evolve.benchmarks.libero_suites import LIBERO_SUITE_TASKS, LIBERO_TASK_SUITE, RLINF_PI05_LIBERO_PROTOCOLS
from robot_auto_evolve.config import Profile
from robot_auto_evolve.protocol import StrictSchemaError
from robot_auto_evolve.provenance import BenchmarkPlan

from .benchmark import LiberoBenchmarkConfig, verify_benchmark_output


@dataclass(frozen=True)
class RLinfPi05LiberoBenchmarkConfig(LiberoBenchmarkConfig):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.benchmark_id != "rlinf_pi05_libero_standard_four_suite_10_per_task_v1":
            raise StrictSchemaError("benchmark_config.benchmark_id: exact RLinf pi0.5 benchmark required")
        if self.model_route != "rlinf_pi05_libero" or self.trials_per_task != 10:
            raise StrictSchemaError("benchmark_config: exact RLinf pi0.5 route and ten trials required")

    def load_profiles(self, project_root: str | Path) -> dict[str, Profile]:
        profiles = super().load_profiles(project_root)
        for suite, profile in profiles.items():
            if (
                profile.environment.suite != suite
                or profile.environment.adapter != "robot_auto_evolve.benchmarks.workers:RLinfPi05LiberoWorker"
                or profile.policy.adapter != "robot_auto_evolve.benchmarks.rlinf_pi05:RLinfPi05LiberoAdapter"
                or len(profile.policy.replicas) < 2
                or any(replica.identity.service_name != "rlinf_pi05_libero" for replica in profile.policy.replicas)
                or any(replica.identity.model_id != "RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT" for replica in profile.policy.replicas)
            ):
                raise StrictSchemaError("benchmark_config: RLinf pi0.5 standard LIBERO profile differs")
        return profiles

    def validate_plan(self, plan: BenchmarkPlan) -> None:
        if not isinstance(plan, BenchmarkPlan) or plan.model_route != self.model_route or len(plan.episodes) != 400:
            raise StrictSchemaError("benchmark_config: exact 400-row RLinf pi0.5 plan required")
        expected_tasks = set(LIBERO_TASK_SUITE)
        if {item.task_id for item in plan.episodes} != expected_tasks:
            raise StrictSchemaError("benchmark_config: all 40 standard LIBERO tasks required")
        by_task = {task: [] for task in expected_tasks}
        for item in plan.episodes:
            by_task[item.task_id].append(item)
            suite = LIBERO_TASK_SUITE[item.task_id]
            protocols: Mapping[str, int] = RLINF_PI05_LIBERO_PROTOCOLS[suite]
            if item.protocol not in protocols or item.horizon != protocols[item.protocol]:
                raise StrictSchemaError("benchmark_config: RLinf pi0.5 protocol or horizon differs")
            expected_protocol = f"rlinf_pi05_libero_{'long' if suite == 'libero_10' else suite.removeprefix('libero_')}_canonical_10_per_task_v1"
            if item.protocol != expected_protocol or item.environment_seed != 7 or item.policy_seed != 7:
                raise StrictSchemaError("benchmark_config: fixed route protocol or RNG seed differs")
        expected_states = {f"init_state_{index:02d}" for index in range(1, 11)}
        for rows in by_task.values():
            if len(rows) != 10 or {item.scenario_id for item in rows} != expected_states:
                raise StrictSchemaError("benchmark_config: scored initial states must be 1 through 10")


def verify_rlinf_pi05_libero_benchmark_output(
    path: str | Path,
    config: RLinfPi05LiberoBenchmarkConfig,
    project_root: str | Path,
) -> dict[str, object]:
    profiles = config.load_profiles(project_root)
    return verify_benchmark_output(
        path,
        plan_validator=config.validate_plan,
        profile_suites=tuple(profiles),
    )
