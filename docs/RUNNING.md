# Running studies

All route launchers use the same operator interface:

```text
ROUTE_LAUNCHER RUN_ID --target-candidates N [resource options] [task options] [--finalize] [--run-transfer]
```

Use only a launcher listed in [ROUTES_AND_TASKS.md](ROUTES_AND_TASKS.md). The generated route specification pins its model revision, tools, profiles, benchmark plan, task rows, horizons, scalar, defaults, and candidate budget.

## Read-only checks

From `/nlp/scr/shgwu/roboAE`:

```bash
./launch/verify_runtime.sh
./launch/routes/libero/xvla_four_suite.sh --help
./launch/routes/libero/xvla_four_suite.sh --list-tasks
```

`--list-tasks` prints the exact accepted task IDs, standard row counts, horizons, protocols, and the pinned `related` preset. It does not start a simulator, model, or Claude process.

Before every GPU run, inspect the live allocation and processes:

```bash
hostname
nproc
nvidia-smi -L
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

Do not interrupt an existing process. A launcher owns and cleans up only the process group it starts.

## Claude setup

The launcher finds `claude` on `PATH`. Set only the credential-directory location:

```bash
export ROBOT_AE_CLAUDE_CREDENTIAL_DIR=/nlp/scr/shgwu/robot_auto_evolve_private/claude_oauth
```

That external directory must have mode `0700` and contain a current-user, regular, singly linked `oauth_token` file with mode `0600`. The token is opened by the trusted local relay. The coding subprocess receives a temporary proxy credential, has no direct host credential access, and runs in `runs/<study-id>/runtime/claude_isolation`.

## Task modes

Full benchmark is the default. Do not pass task flags:

```bash
./launch/routes/libero/xvla_four_suite.sh full_001 --target-candidates 1
```

For the pinned related-task split, use:

```bash
./launch/routes/libero/xvla_four_suite.sh transfer_001 \
  --task-preset related \
  --target-candidates 1
```

For a new, separately audited split, repeat both flags:

```bash
./launch/routes/libero/xvla_four_suite.sh custom_001 \
  --evolve-task pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate \
  --transfer-task pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate \
  --target-candidates 1
```

Both explicit sets must be nonempty and disjoint. VLABench additionally rejects task units whose qualified IDs differ but whose underlying semantic task name overlaps. Explicit task filtering keeps the exact standard rows for each selected task; it never regenerates trials or changes horizons. Arbitrary disjoint tasks are not automatically a meaningful related-transfer experiment.

## Candidate target, resume, and freeze

`--target-candidates N` is the total number of completed proposals requested for the study, not the number to add in this invocation. The baseline is separate. Accepted and rejected proposals both count; a failed or interrupted proposal does not count.

Start with one proposal:

```bash
./launch/routes/libero/xvla_four_suite.sh full_001 \
  --target-candidates 1 \
  --gpu-ids 0,1 \
  --workers-per-gpu 4 \
  --port-offset 0
```

Repeat the identical command to resume an interruption. To continue later, increase only the target, for example to `2`. The route budget is 30.

Freeze only after the target has completed:

```bash
./launch/routes/libero/xvla_four_suite.sh full_001 \
  --target-candidates 1 \
  --gpu-ids 0,1 \
  --workers-per-gpu 4 \
  --port-offset 0 \
  --finalize
```

Finalization is irreversible for that study ID. It creates no new proposal and refuses a target above the completed count.

For a related-task study, transfer requires finalization and runs baseline versus frozen scaffold exactly once on the held-out rows:

```bash
./launch/routes/libero/xvla_four_suite.sh transfer_001 \
  --task-preset related \
  --target-candidates 1 \
  --gpu-ids 0,1 \
  --workers-per-gpu 4 \
  --port-offset 0 \
  --finalize \
  --run-transfer
```

Transfer outcomes never enter proposal generation or acceptance.

## GPUs, workers, servers, and rendering

The default profile uses two GPUs. `--gpu-ids` is a sorted, unique list with at least two entries. The runtime creates one policy server per selected GPU.

```bash
--gpu-ids 0,1
--gpu-ids 0,1,2,3
```

`--workers-per-gpu N` sets simulator concurrency per selected GPU. Total workers equal `GPU count × workers per GPU`. Most routes default to four workers per GPU; RLDX + RoboCasa365 and the gated RoboTwin route default to two. Each route's exact default and shared tool-server count are listed in [ROUTES_AND_TASKS.md](ROUTES_AND_TASKS.md). Shared language and perception tool servers keep their checked placement on the first two selected GPUs; they are not duplicated on every GPU.

`--render-gpu-ids` gives one rendering assignment per policy replica. It defaults to the selected GPU list, may repeat a selected GPU, and cannot name a GPU outside the pool:

```bash
--gpu-ids 0,1 --render-gpu-ids 1,1
```

A single four-GPU study is therefore:

```bash
./launch/routes/libero/xvla_four_suite.sh full_4gpu_001 \
  --target-candidates 1 \
  --gpu-ids 0,1,2,3 \
  --workers-per-gpu 4 \
  --port-offset 0
```

On a four-GPU job, two production-shaped studies can run in parallel on two GPUs each. Use disjoint GPU lists and port ranges:

```bash
./launch/routes/libero/xvla_four_suite.sh xvla_001 \
  --target-candidates 1 --gpu-ids 0,1 --workers-per-gpu 4 --port-offset 0
```

```bash
./launch/routes/libero/lerobot_pi05_four_suite.sh pi05_001 \
  --target-candidates 1 --gpu-ids 2,3 --workers-per-gpu 4 --port-offset 1000
```

Run those commands in separate terminals. A study's GPU list, render assignment, workers per GPU, port offset, task split, and pinned inputs are immutable after its first materialization. Resume with the same values.

## Run layout

For route `xvla_libero` and run ID `full_001`, all owned state is below:

```text
runs/xvla_libero_full_001/
```

Important paths are:

```text
study_request.json                 immutable route, task, plan, and resource request
runtime/profile.json               primary materialized runtime profile
runtime/profiles/*.json            suite or cell runtime profiles
runtime/profile_materialization.json
runtime/claude_isolation/          run-local coding sandbox state
runtime/invocations/               disposable service and evaluator scratch
invocations/                       immutable invocation requests, results, and system captures
evolution/baseline/                seed-scaffold result
evolution/candidates/0001/         proposal, benchmark result, evidence, and decision
evolution/state.json               checked resume state
evolution/frozen/                  finalized scaffold
evolution/transfer/                optional held-out baseline-versus-frozen result
```

The runtime directory does not contain another copy of model weights, environments, datasets, or upstream repositories. Those shared read-only artifacts remain at the locations in `runtime_paths.json`. Run-local runtime state exists so concurrent studies and resumes do not share ports, logs, simulator scratch, service state, or Claude isolation.

## Comparison labels

The six canonical aggregate routes run all 40 LIBERO tasks or all 80 LIBERO-Pro task units with one scaffold at every candidate. Their scores match the pinned benchmark protocol, but the scaffold was optimized on that benchmark and uses extra frozen tools. Report them as benchmark-adapted agent scores.

A suite or cell slice is directly interpretable only for that slice. The convenience batch launchers start independent studies that may freeze different scaffolds; do not average them and call the result one shared four-suite or eight-cell agent.

A related-task result evaluates only its selected standard rows and answers transfer from the evolve tasks to held-out related tasks. It is not a full-benchmark score.

## Safe dry preparation

The hidden `--prepare-only` option resolves the route, task mode, resource profile, immutable plan hashes, and run-local profile files without starting services or Claude. Use it only with a disposable new run ID:

```bash
./launch/routes/libero/xvla_four_suite.sh drycheck_001 \
  --target-candidates 1 \
  --gpu-ids 0,1 \
  --workers-per-gpu 4 \
  --port-offset 0 \
  --prepare-only
```

It deliberately creates `runs/xvla_libero_drycheck_001/`; do not reuse that ID for a differently configured study.
