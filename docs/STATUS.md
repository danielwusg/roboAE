# Route status

There are 44 implemented routes in this clean tree. Forty-two have a complete benchmark plan and evaluator. Six of those 42 are canonical aggregate routes: five policy families over all four standard LIBERO suites and RLinf pi0.5 over all eight LIBERO-Pro cells. The 20 LIBERO suite routes and eight LIBERO-Pro cell routes remain focused slices. Two integration routes intentionally fail before production because their complete benchmark is not ready.

“Implemented” means the environment adapter, policy service, enabled tools, seed scaffold, and evaluator path exist and have retained development evidence. It does not mean a full benchmark or Claude-driven evolution run has been completed.

| Family | Implemented routes | Full benchmark state |
|---|---:|---|
| X-VLA + LIBERO | 1 canonical + 4 slices | Canonical route ready with 2,000 rows; slices have 500 rows each |
| LeRobot pi0.5 + LIBERO | 1 canonical + 4 slices | Canonical route ready with 400 rows; slices have 100 rows each |
| MolmoAct2 Base + LIBERO | 1 canonical + 4 slices | Canonical route ready with 2,000 rows; slices have 500 rows each |
| MolmoAct2 Think + LIBERO | 1 canonical + 4 slices | Canonical route ready with 2,000 rows; slices have 500 rows each |
| RLinf pi0.5 + standard LIBERO | 1 canonical + 4 slices | Canonical route ready with 400 rows; slices have 100 rows each |
| RLinf pi0.5 + LIBERO-Pro | 1 canonical + 8 slices | Canonical route ready with 800 rows; slices have 100 rows each |
| X-VLA + SimplerEnv Google VA, Google VM, WidowX VM | 3 | Ready with protocol caveats in [BENCHMARKS.md](BENCHMARKS.md) |
| OpenVLA base + SimplerEnv Google VA, Google VM | 2 | Ready as project benchmark routes, not OpenVLA paper headlines |
| X-VLA + VLABench tracks 1–4 | 1 | Ready; 400 rows |
| RLDX-1 + RoboCasa365 Target-50 | 1 | Ready; 2,500 rows |
| SmolVLA substitute + public RoboCerebra | 1 | Ready; 600-row public-substitute protocol, not paper-comparable |
| X-VLA + CALVIN ABC to D | 1 | Integration only; full X-VLA 1,000-sequence evaluator missing |
| X-VLA + RoboTwin 2.0 | 1 | Integration only; real expert-filtered 5,000-row plan must be prepared |

The exact 44 launchers, model revisions, tools, tasks, trials, horizons, related-transfer presets, and route-specific blockers are in [ROUTES_AND_TASKS.md](ROUTES_AND_TASKS.md).

## Not-ready or unavailable combinations

These are recorded so that a public component is not mistaken for a runnable route.

| Combination | What exists | What is still missing |
|---|---|---|
| MolmoBot-DROID + RoboLab-120 | Exact two-replica policy startup | A completed Isaac/RoboLab simulator episode and explicit NVIDIA Omniverse EULA acceptance |
| MolmoAct2-DROID + RoboLab-120 | Exact two-replica stateful policy protocol | Same simulator and EULA gate |
| OpenPI pi0.5 joint-position + RoboLab-120 | Exact benchmark-native checkpoint and policy wiring | Final GPU policy seal plus the simulator and EULA gate |
| OpenPI pi0-FAST joint-position + RoboLab-120 | Exact benchmark-native checkpoint and policy wiring | Final GPU policy seal plus the simulator and EULA gate |
| DreamZero-DROID + RoboLab-120 | Source and checkpoint pinned | Local backend, policy protocol seal, simulator, and EULA gate |
| Generic OpenPI DROID velocity policies + current RoboLab evaluator | Public checkpoints | Controller contracts conflict: velocity outputs cannot be relabeled as absolute joint positions |
| Exact VoLoAgent + RoboLab or RoboVoLo | Public descriptions and several independent components | Agent code, planner state/prompts, complete checkpoint set, and for RoboVoLo the executable benchmark package |
| Exact Harness-VLA | Public policy/environment pieces used by several implemented routes | Agent code, memory bootstrap, demonstrations, prompts, and exact RoboTwin C2R artifacts |
| LingBot-VLA-2.0 + RoboTwin | Public base model and source | A released task-post-trained RoboTwin checkpoint |
| X-VLA + NAVSIM | X-VLA source and other environment checkpoints | A released NAVSIM policy checkpoint and evaluator |
| Paper OpenVLA-OFT + RoboCerebra | Public benchmark and SmolVLA substitute | The paper policy checkpoint and exact paper protocol assets |
| RoboVoLo benchmark | Reported task and trial counts | Public tasks, simulator package, resets, and seed table |
| CaP-X local baseline | Paper and source description | Exact 39-task local manifest, provider settings, reset/seed plan, and API integration |
| TiPToP local baseline | Paper and source description | Exact 28-task local manifest, provider settings, reset/seed plan, and API integration |

GraspGen is a tool rather than a model–environment route. It remains disabled wherever the fair observation lacks metric depth, calibration, and a matching motion executor. Every enabled route still starts from the same VoLo/Harness-like scaffold: a frozen route-specific low-level policy plus frozen language, vision, pointing, detection, and segmentation services where the route profile marks them available.
