#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_paths=${ROBOT_AE_RUNTIME_PATHS:-"$root/runtime_paths.json"}
verify_python=${ROBOT_AE_VERIFY_PYTHON:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_root"] + "/core/bin/python")' "$runtime_paths")}

exec env -u PYTHONHOME -u VIRTUAL_ENV \
  HOME=/nonexistent \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$root/src" \
  ROBOT_AE_CLEAN_SRC="$root/src" \
  ROBOT_AE_PROJECT_ROOT="$root" \
  ROBOT_AE_RUNTIME_PATHS="$runtime_paths" \
  "$verify_python" -m robot_auto_evolve.runtime_paths verify --project-root "$root" "$@"
