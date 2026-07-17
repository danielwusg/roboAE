#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "$ROOT/launch/route_group.sh" 'xvla_libero_suite_slice_batch' "$@"
