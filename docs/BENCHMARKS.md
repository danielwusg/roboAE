# Benchmark protocols and comparability

The default launch mode puts the complete route benchmark inside the coding-agent loop. The baseline and every completed candidate run every row in the route plan exactly once. The coding agent receives every episode outcome plus bounded, fair trajectory evidence, and a candidate is accepted only when the route's declared scalar score strictly increases. There is no second evaluation pass in this mode.

This makes every iteration a result under the stated task, trial, horizon, controller, and metric protocol. It also means the scaffold was adapted to that benchmark. Report both facts. A number is directly comparable at the protocol level only when the task rows and aggregation below match; it is not evidence of untouched held-out generalization.

## Implemented benchmark plans

| Route family | Complete plan used by one route | Horizon or action schedule | Primary score | Comparison boundary |
|---|---:|---|---|---|
| X-VLA + full LIBERO | 40 tasks × 50 trials = 2,000 | 800 steps for Spatial, Object, and Goal; 900 for Long | Equal-suite, equal-task success | Canonical `xvla_libero` route: one shared scaffold is evolved on all four suites at every candidate. |
| LeRobot pi0.5 + full LIBERO | 40 tasks × 10 trials = 400 | 280, 280, 300, or 520 steps; predict 50 and execute 10 | Equal-suite, equal-task success | Canonical `pi05_libero` route; this is the pinned LeRobot checkpoint protocol, not the OpenPI or Harness policy protocol. |
| MolmoAct2 Base or Think + full LIBERO | 40 tasks × 50 trials = 2,000 per checkpoint | 280, 280, 300, or 520 steps; predict and execute 10 | Equal-suite, equal-task success | Canonical `molmoact2_libero` and `molmoact2_think_libero` routes. Base and Think remain separate agents. |
| RLinf pi0.5 + full standard LIBERO | 40 tasks × states 1–10 = 400 | 220, 280, 300, or 520 steps; predict 10 and execute 5 | Equal-suite, equal-task success | Canonical `rlinf_pi05_libero` route; it uses the public full-shot policy, not the unreleased Harness memory agent. |
| RLinf pi0.5 + full LIBERO-Pro | 80 namespaced tasks × states 1–10 = 800 | 220, 280, 300, or 520 steps; predict 10 and execute 5 | Equal-cell, equal-task success | Canonical `rlinf_pi05_libero_pro` route: one shared scaffold is evolved across all eight task/swap cells. |
| One LIBERO suite slice | 10 tasks × 10 or 50 trials = 100 or 500 | Same schedule as its canonical parent | Equal-task success within the suite | Useful for a focused study. Four independently evolved slices are not one shared four-suite agent. |
| One LIBERO-Pro cell slice | 10 tasks × states 1–10 = 100 | Same schedule as its canonical parent | Equal-task success within the cell | Useful for a focused study. Eight independently evolved slices are not one shared eight-cell agent. |
| X-VLA + SimplerEnv Google Variant Aggregation | 1,992 rows across five tasks | 160, 226, or 400 actions by task | Task-macro success | Complete local five-task grid. The X-VLA paper headline reports four named families and excludes close-drawer, so the five-task macro is not that headline. |
| X-VLA + SimplerEnv Google Visual Matching | 864 rows across five tasks | 160, 226, or 400 actions by task | Task-macro success | Same five-task versus four-family boundary as Variant Aggregation. |
| X-VLA + SimplerEnv WidowX Visual Matching | 4 tasks × 24 rows = 96 | 1,200-step limit | Task-macro success | Uses the disclosed absolute-controller correction; it is not the untouched released client default. |
| OpenVLA base + SimplerEnv Google Variant Aggregation | 1,992 rows across five tasks | 80, 113, or 200 actions by task | Task-macro success | Complete project grid for the base checkpoint, not an OpenVLA paper headline. |
| OpenVLA base + SimplerEnv Google Visual Matching | 864 rows across five tasks | 80, 113, or 200 actions by task | Task-macro success | Complete project grid for the base checkpoint, not an OpenVLA paper headline. |
| X-VLA + VLABench tracks 1–4 | 40 track–task units × 10 configurations = 400 | 100 or 200 policy steps by task | Equal-track, then equal-task progress score | There are 12 unique semantic task names because names repeat across tracks. Success and intention are retained secondary metrics. |
| RLDX-1 + RoboCasa365 Target-50 | 50 tasks × 50 trials = 2,500 | Task-specific 300–2,900 steps; predict 16 and execute 8 | Equal weight over atomic-seen, composite-seen, and composite-unseen group means | Matches the public pinned RLDX Target-50 schedule, not Harness's smaller memory-assisted schedule. |
| SmolVLA substitute + public RoboCerebra | 60 logical cases × 10 trials = 600 | 150 × substep count minus 15 settle steps | Equal-condition, equal-case success | Launchable public substitute only. The paper OpenVLA-OFT checkpoint is unavailable; the public data has 45 initialization files, and this evaluator reconstructs starts from demonstrations and changes several released behaviors. Do not call it a paper reproduction. |

Every exact task ID, row count, horizon, protocol label, model revision, and launcher is generated in [ROUTES_AND_TASKS.md](ROUTES_AND_TASKS.md). The JSON plan beside each route is the executable authority.

## Implemented integration routes without a launchable full plan

| Route | Intended complete setup | Current blocker |
|---|---|---|
| X-VLA + CALVIN ABC to D | 1,000 deterministic domain-D sequences, five subtasks per sequence; average completed chain and success-at-1 through success-at-5 | The X-VLA-specific 1,000-sequence evaluator is not implemented. The released X-VLA client also uses 720 steps per subtask and one reset per sequence, unlike upstream CALVIN's 360 steps and per-subtask reset. The launcher fails before starting services. |
| X-VLA + RoboTwin 2.0 `demo_clean` | 50 tasks × 100 expert-success-filtered seeds = 5,000; equal-task success | The short integration route is sealed, but the real expert seed-preparation pass has not produced the immutable 5,000-row plan. The launcher fails before starting services until that plan is prepared and pinned. |

## Related-task transfer mode

Passing `--task-preset related` changes the study to the route's pinned related-task transfer split. The exact evolve and held-out task IDs and source EpisodePlan hashes are listed in [ROUTES_AND_TASKS.md](ROUTES_AND_TASKS.md). Repeated `--evolve-task` and `--transfer-task` flags remain available for a newly audited split; both sets must be nonempty, valid, and disjoint. Arbitrary disjoint tasks do not by themselves support a related-task transfer claim. The selected tasks keep every standard scenario, seed, trial, horizon, controller, and success rule from the route plan.

Only evolution-task outcomes and bounded trajectories are available while the scaffold changes. After the requested candidates are complete, finalization freezes the chosen scaffold. Transfer then runs exactly twice on the held-out rows: once with the original seed scaffold and once with the frozen evolved scaffold. Transfer never affects candidate acceptance.

These results answer whether a mechanism learned on related tasks transfers. They are not the full benchmark headline because only selected task units are evaluated.

## Result validity

A result is complete only when every planned episode ID appears exactly once, every service and artifact identity matches, and no simulator, renderer, policy, or tool failure is counted as task failure. The run stores the exact route, profile, plan, task-selection, resource, and scaffold hashes. Resume rejects changes to any frozen input.
