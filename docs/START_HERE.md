# Start here

This repository runs coding-agent-driven evolution of a frozen multi-model robot agent. The coding agent edits only the scaffold that decides when and how to call the frozen low-level policy, language model, vision model, detector, segmenter, and pointing model. It does not train those models.

There are 44 implemented routes. Forty-two have complete benchmark plans. The six primary full-benchmark routes keep one shared evolving scaffold across the complete environment benchmark:

| Route ID | Model and environment | Standard rows | Scalar score | Launcher |
|---|---|---:|---|---|
| `xvla_libero` | X-VLA + all four LIBERO suites | 2,000 | Equal-suite, equal-task success | `launch/routes/libero/xvla_four_suite.sh` |
| `pi05_libero` | LeRobot pi0.5 + all four LIBERO suites | 400 | Equal-suite, equal-task success | `launch/routes/libero/lerobot_pi05_four_suite.sh` |
| `molmoact2_libero` | MolmoAct2 Base + all four LIBERO suites | 2,000 | Equal-suite, equal-task success | `launch/routes/libero/molmoact2_base_four_suite.sh` |
| `molmoact2_think_libero` | MolmoAct2 Think + all four LIBERO suites | 2,000 | Equal-suite, equal-task success | `launch/routes/libero/molmoact2_think_four_suite.sh` |
| `rlinf_pi05_libero` | RLinf pi0.5 + all four LIBERO suites | 400 | Equal-suite, equal-task success | `launch/routes/libero/rlinf_pi05_four_suite.sh` |
| `rlinf_pi05_libero_pro` | RLinf pi0.5 + all eight LIBERO-Pro cells | 800 | Equal-cell, equal-task success | `launch/routes/libero_pro/rlinf_pi05_eight_cell.sh` |

The other 36 launch-ready routes cover focused LIBERO suites, focused LIBERO-Pro cells, SimplerEnv, VLABench, RoboCasa365, and the public RoboCerebra substitute. X-VLA + CALVIN and X-VLA + RoboTwin are integrated but gated because their complete benchmark evaluators or prepared plans are missing. See [STATUS.md](STATUS.md) and [ROUTES_AND_TASKS.md](ROUTES_AND_TASKS.md).

## Before a run

Enter the clean repository, inspect the machine, and do not stop an existing process:

```bash
cd /nlp/scr/shgwu/roboAE
hostname
nproc
nvidia-smi -L
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
./launch/verify_runtime.sh
```

The clean repository reuses the locked Conda environments, model cache, simulators, datasets, and assets recorded in `runtime_paths.json`; it does not copy them. Run-owned files stay inside `runs/<study-id>/`. See [RUNTIME.md](RUNTIME.md).

Check a route before spending Claude usage:

```bash
./launch/routes/libero/xvla_four_suite.sh --help
./launch/routes/libero/xvla_four_suite.sh --list-tasks
```

## Claude credential

`claude` must be on `PATH`. The online relay reads one raw OAuth token from a private directory outside the repository and every run directory. Do not put the token itself in an environment variable or command line.

```bash
export ROBOT_AE_CLAUDE_CREDENTIAL_DIR=/nlp/scr/shgwu/robot_auto_evolve_private/claude_oauth
install -d -m 700 "$ROBOT_AE_CLAUDE_CREDENTIAL_DIR"
install -m 600 /path/to/your/raw/oauth_token "$ROBOT_AE_CLAUDE_CREDENTIAL_DIR/oauth_token"
claude --version
```

The directory must be owned by you with mode `0700`; `oauth_token` must be a regular, singly linked, current-user file with mode `0600`.

## Start a standard full-benchmark evolution

No task flags means the complete pinned route plan is evaluated for the baseline and every candidate:

```bash
./launch/routes/libero/xvla_four_suite.sh full_001 \
  --target-candidates 30 \
  --gpu-ids 0,1 \
  --workers-per-gpu 4 \
  --port-offset 0
```

`--target-candidates 30` means 30 completed scaffold proposals in total, excluding the baseline. Accepted and rejected proposals both count. Repeat the same command after an interruption. You may first target `1`, inspect it, then resume with `2`, up to the route budget of 30.

After the requested candidates finish, freeze the chosen scaffold:

```bash
./launch/routes/libero/xvla_four_suite.sh full_001 \
  --target-candidates 30 \
  --gpu-ids 0,1 \
  --workers-per-gpu 4 \
  --port-offset 0 \
  --finalize
```

The study is stored at `runs/xvla_libero_full_001/`. Full-benchmark candidate scores use the standard task rows, trials, horizons, and scalar, so they are comparable at the protocol level. They must still be labeled as benchmark-adapted agent results, not untouched held-out results or raw-policy scores.

## Run the pinned related-task transfer study

Every launch-ready route has a reviewed preset derived from its pinned legacy EpisodePlan:

```bash
./launch/routes/libero/xvla_four_suite.sh transfer_001 \
  --task-preset related \
  --target-candidates 30 \
  --gpu-ids 0,1 \
  --workers-per-gpu 4 \
  --port-offset 0
```

After evolution, freeze and evaluate transfer:

```bash
./launch/routes/libero/xvla_four_suite.sh transfer_001 \
  --task-preset related \
  --target-candidates 30 \
  --gpu-ids 0,1 \
  --workers-per-gpu 4 \
  --port-offset 0 \
  --finalize \
  --run-transfer
```

Only evolve-task evidence can change the scaffold. Held-out tasks run after freezing, once with the original scaffold and once with the frozen scaffold. This is a transfer result, not a full-benchmark headline. The exact preset task IDs are in [ROUTES_AND_TASKS.md](ROUTES_AND_TASKS.md).

For all lifecycle, GPU, worker, port, resume, result-layout, and two-run examples, read [RUNNING.md](RUNNING.md).
