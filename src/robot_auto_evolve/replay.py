from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robot_auto_evolve.config import Profile
from robot_auto_evolve.evaluation.simulator import SimulatorProcess
from robot_auto_evolve.evolution.profile_evaluator import simulator_timeouts, success_protocol
from robot_auto_evolve.protocol import CanonicalActionChunk, StrictSchemaError
from robot_auto_evolve.provenance import EpisodeKey
from robot_auto_evolve.runtime import resolve_profile_launch_paths
from robot_auto_evolve.runtime_paths import RuntimePaths, project_root_from_package
from robot_auto_evolve.study_runner import _simulator_source_key


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_run_root(episode_dir: Path) -> Path:
    for parent in [episode_dir, *episode_dir.parents]:
        if (parent / "runtime" / "profiles").is_dir() and (parent / "study_request.json").is_file():
            return parent
    raise FileNotFoundError(
        f"no run directory above {episode_dir}: expected a folder holding study_request.json and runtime/profiles/"
    )


def load_profile(run_root: Path, task_id: str, project_root: Path) -> Profile:
    profiles = sorted((run_root / "runtime" / "profiles").glob("*.json"))
    if not profiles:
        raise FileNotFoundError(f"no runtime profile under {run_root / 'runtime' / 'profiles'}")
    if len(profiles) > 1 and "::" in task_id:
        suite = task_id.split("::", 1)[0]
        named = [path for path in profiles if path.stem == suite]
        if named:
            profiles = named
    return Profile.load(profiles[0], project_root=project_root)


def read_actions(trace: Path) -> list[list[list[float]]]:
    actions = []
    with trace.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index == 0:
                continue
            row = json.loads(line)
            if row.get("action") is not None:
                actions.append(row["action"]["values"])
    return actions


def replay(
    episode_dir: Path,
    output_dir: Path,
    *,
    render_gpu: int,
    action_source: Path | None,
    max_steps: int,
    save_depth: bool,
) -> dict[str, Any]:
    project_root = project_root_from_package()
    episode_dir = Path(episode_dir).resolve()
    manifest = _load_json(episode_dir / "episode.json")
    key = EpisodeKey.from_mapping(manifest["key"])
    run_root = find_run_root(episode_dir)
    profile = load_profile(run_root, key.task_id, project_root)

    values = read_actions(episode_dir / "trace.jsonl" if action_source is None else action_source)
    if not values:
        raise StrictSchemaError("no actions found; expected 'action' entries in the trace")
    if max_steps > 0:
        values = values[:max_steps]

    launch = resolve_profile_launch_paths(profile, project_root, RuntimePaths.load(project_root).environment_root)
    start_timeout_s, call_timeout_s = simulator_timeouts(profile.environment.suite)
    runtime_dir = run_root / "runtime" / "replay" / episode_dir.name
    if runtime_dir.exists():
        raise FileExistsError(f"replay scratch already exists, remove it first: {runtime_dir}")
    runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    full_horizon = success_protocol(profile) == "full_horizon_final_success"
    simulator = SimulatorProcess(
        launch["simulator_python"],
        profile,
        key,
        physical_gpu_id=render_gpu,
        runtime_dir=runtime_dir,
        source_root=launch[_simulator_source_key(profile.environment.suite)],
        start_timeout_s=start_timeout_s,
        call_timeout_s=call_timeout_s,
    )
    from PIL import Image

    success = False
    executed = 0
    completed = False
    try:
        simulator.start()
        simulator.reset()
        for index, row in enumerate(values):
            observation = simulator.observe()
            for name in sorted(observation.cameras):
                camera = observation.cameras[name]
                Image.fromarray(camera.rgb, mode="RGB").save(
                    output_dir / f"{name}-{observation.step_index:08d}.png"
                )
                if save_depth and camera.depth_m is not None:
                    np.save(output_dir / f"{name}-depth-{observation.step_index:08d}.npy", camera.depth_m)
            if not full_horizon and simulator.private_success():
                success = True
                break
            chunk = CanonicalActionChunk(
                request_id=f"replay-{index}",
                session_id=key.artifact_id(),
                start_step=observation.step_index,
                spec=profile.policy.action_spec,
                values=np.asarray(row, dtype=np.float32),
                execution_count=profile.policy.execution_count,
            )
            simulator.apply(profile.validate_agent_action_chunk(chunk))
            executed += 1
            if not full_horizon and simulator.private_success():
                success = True
                break
        if full_horizon:
            success = simulator.private_success()
        metrics = simulator.private_metrics()
        completed = True
    finally:
        simulator.close(force=True)
        if completed:
            shutil.rmtree(runtime_dir, ignore_errors=True)

    report = {
        "episode_id": key.artifact_id(),
        "task_id": key.task_id,
        "actions_applied": executed,
        "actions_available": len(values),
        "success": bool(success),
        "recorded_success": manifest.get("success"),
        "private_metrics": metrics,
        "pictures": str(output_dir),
    }
    (output_dir / "replay.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="robot_auto_evolve.replay",
        description=(
            "Re-run one recorded episode in the simulator and save every camera picture. "
            "Reads the task, scene and seeds from episode.json and the actions from trace.jsonl, "
            "so the run is the same one that was recorded. Pass --actions to replay a different "
            "action sequence instead and see what would have happened. No policy and no other "
            "model is loaded."
        ),
    )
    parser.add_argument("--episode", type=Path, required=True, help="an episode folder holding episode.json and trace.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="where to write the pictures")
    parser.add_argument("--render-gpu", type=int, default=1, help="which GPU renders")
    parser.add_argument("--actions", type=Path, help="a trace.jsonl to take the actions from instead")
    parser.add_argument("--steps", type=int, default=0, help="stop after this many steps (0 means all)")
    parser.add_argument("--depth", action="store_true", help="also save each camera's depth as .npy")
    args = parser.parse_args(argv)
    report = replay(
        args.episode,
        args.out,
        render_gpu=args.render_gpu,
        action_source=args.actions,
        max_steps=args.steps,
        save_depth=args.depth,
    )
    json.dump(report, sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
