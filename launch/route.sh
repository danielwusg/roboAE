#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUNTIME_PATHS=${ROBOT_AE_RUNTIME_PATHS:-"$ROOT/runtime_paths.json"}
if [[ -n ${ROBOT_AE_CORE_PYTHON:-} ]]; then
  PYTHON=$ROBOT_AE_CORE_PYTHON
elif [[ -n ${ROBOT_AE_ENV_ROOT:-} ]]; then
  PYTHON=$ROBOT_AE_ENV_ROOT/core/bin/python
else
  PYTHON=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_root"] + "/core/bin/python")' "$RUNTIME_PATHS")
fi
[[ -x "$PYTHON" ]] || { echo "core Python is missing; set ROBOT_AE_CORE_PYTHON or ROBOT_AE_ENV_ROOT" >&2; exit 2; }
export PYTHONPATH=$ROOT/src
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export ROBOT_AE_CLEAN_SRC=$ROOT/src
export ROBOT_AE_PROJECT_ROOT=$ROOT
export ROBOT_AE_RUNTIME_PATHS=$RUNTIME_PATHS
exec "$PYTHON" -m robot_auto_evolve.operator_cli route --project-root "$ROOT" --route "$1" "${@:2}"
