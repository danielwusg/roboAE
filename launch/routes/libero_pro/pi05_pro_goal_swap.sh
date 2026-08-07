#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec "$ROOT/launch/route.sh" 'pi05_libero_pro_goal_swap' "$@"
