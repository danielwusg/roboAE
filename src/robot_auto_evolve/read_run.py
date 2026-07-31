
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _episode_dirs(phase_dir: Path) -> list[Path]:
    if not phase_dir.is_dir():
        return []
    return sorted(p for p in phase_dir.glob("**/episodes/benchmark-*") if p.is_dir())


def _tally(phase_dir: Path) -> tuple[int, int]:
    done = succ = 0
    for ep in _episode_dirs(phase_dir):
        manifest = _load(ep / "episode.json")
        if manifest is None:
            continue
        done += 1
        if manifest.get("success") is True:
            succ += 1
    return done, succ


def _score(phase_dir: Path):
    result = _load(phase_dir / "benchmark_result.json")
    if isinstance(result, dict) and isinstance(result.get("scalar"), dict):
        return result["scalar"].get("value")
    return None


def _fmt_phase(name: str, phase_dir: Path, committed: bool, extra: str = "") -> str:
    done, succ = _tally(phase_dir)
    score = _score(phase_dir)
    score_s = f"score={score:.4f}" if isinstance(score, (int, float)) else "score=(pending)"
    tag = "committed" if committed else "in-progress"
    return f"  {name:<16} {score_s}  ({done} episodes done, {succ} success)  [{tag}]{extra}"


def read_run(target: str, project_root: Path) -> str:
    run_dir = Path(target)
    if not run_dir.is_dir():
        run_dir = project_root / "runs" / target
    if not run_dir.is_dir():
        return f"no run found at {target} (tried {run_dir})"

    lines: list[str] = []
    req = _load(run_dir / "study_request.json") or {}
    route = req.get("route_id", "?")
    metric = req.get("scalar_metric", "?")
    evolve_ep = (req.get("effective_plan") or {}).get("evolve_episode_count", "?")
    lines.append(f"run:   {run_dir.name}")
    lines.append(f"route: {route}    metric: {metric}    evolve episodes/eval: {evolve_ep}")

    evo = run_dir / "evolution"
    state = _load(evo / "state.json")
    if state is None:
        lines.append("phase: BASELINE (no state.json yet -> baseline still running or not started)")
    else:
        lines.append(
            f"phase: {state.get('phase')}    incumbent: {state.get('incumbent')}    "
            f"next_candidate: {state.get('next_candidate')}"
        )

    if (evo / "baseline").is_dir():
        lines.append(_fmt_phase("baseline", evo / "baseline", True))
    elif (evo / ".baseline-staging").is_dir():
        lines.append(_fmt_phase("baseline", evo / ".baseline-staging", False, "  <- running now"))

    cand_root = evo / "candidates"
    if cand_root.is_dir():
        for cand in sorted(p for p in cand_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            decision = _load(cand / "decision.json") or {}
            verdict = ""
            if decision:
                verdict = (
                    f"  -> {'ACCEPTED' if decision.get('accepted') else 'rejected'}"
                    f" (cand={decision.get('candidate')}, incumbent={decision.get('incumbent')},"
                    f" delta={decision.get('delta')})"
                )
            lines.append(_fmt_phase(f"candidate {cand.name}", cand, True, verdict))
        for staging in sorted(p for p in cand_root.iterdir() if p.is_dir() and p.name.startswith(".")):
            idx = staging.name.lstrip(".").split("-")[0]
            cc = "CC-revision started" if (staging / "revision_prompt.txt").is_file() else "starting"
            has_eval = (staging / "benchmark").is_dir()
            note = "  <- running now"
            lines.append(_fmt_phase(f"candidate .{idx}", staging, False, f"  ({cc}){note}"))

    failures = sorted(p.name for p in (evo / "failures").glob("*")) if (evo / "failures").is_dir() else []
    lines.append(f"  failures/broken: {failures if failures else 'none'}")

    for name in ("metrics.csv", "progress.log"):
        f = evo / name
        if f.is_file():
            body = f.read_text(encoding="utf-8", errors="replace").splitlines()
            lines.append(f"--- {name} (last {min(len(body), 12)} of {len(body)} lines) ---")
            lines.extend("  " + ln for ln in body[-12:])

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="read_run", description="Read an evolution run's status (read-only).")
    parser.add_argument("target", help="study_id (under <project-root>/runs/) or a run directory path")
    parser.add_argument(
        "--project-root",
        default=os.environ.get("ROBOT_AE_PROJECT_ROOT", "/nlp/scr/shgwu/roboAE"),
        help="project root containing runs/ (default: $ROBOT_AE_PROJECT_ROOT or /nlp/scr/shgwu/roboAE)",
    )
    args = parser.parse_args(argv)
    print(read_run(args.target, Path(args.project_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
