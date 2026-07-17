# Experiment design

## Default: full benchmark in every iteration

Each route defaults to its exact standard benchmark plan. The baseline and every candidate run the complete task, scenario, seed, trial, horizon, controller, and metric schedule declared for that route. The coding agent receives the complete benchmark outcomes and bounded trajectory evidence from the current incumbent, revises only `scaffold.py`, and the candidate replaces the incumbent only when the route's benchmark score strictly improves.

There is no second evaluation in this mode. The benchmark is intentionally inside the optimization loop. Every iteration therefore reports a score under the standard benchmark protocol, while the run metadata also states that the scaffold was adapted to that benchmark.

`--target-candidates N` is the total number of completed scaffold proposals, excluding the baseline. Accepted and rejected proposals both count. Failed or interrupted proposals do not count and resume retries the same proposal number. Every route currently caps the total at 30.

Finalization never creates candidates. Complete the requested candidates first, then rerun with the same total and `--finalize`.

## Optional related-task transfer

Every launch-ready route has a pinned `--task-preset related` split derived from its immutable legacy EpisodePlan. The generated inventory lists the exact evolve tasks, held-out tasks, and source plan hashes. Launchers also accept repeated `--evolve-task TASK_ID` and `--transfer-task TASK_ID` arguments for a newly audited split. With no task arguments or preset, the route uses the full benchmark and no transfer split. When explicit task arguments are supplied, evolution and transfer task sets must both be present, valid for the route, nonempty, and disjoint. Arbitrary disjoint task lists are not automatically a meaningful related-task study.

The selected tasks retain their standard benchmark scenarios, trials, seeds, horizons, controller, and success rule. The coding agent sees only evolution-task results. Transfer tasks are evaluated only after finalization, once with the baseline scaffold and once with the frozen evolved scaffold.

The first invocation records the resolved task lists and exact filtered plan hashes. Resume rejects any task, resource, checkpoint, or plan change.

## Result labels

The immutable `study_request.json` records `mode` as `full_benchmark` or `related_transfer`. Results from the two modes must not be merged. Integration smokes are stored separately and are not benchmark results.
