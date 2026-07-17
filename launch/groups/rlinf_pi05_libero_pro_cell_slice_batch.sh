#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "$ROOT/launch/route_group.sh" 'rlinf_pi05_libero_pro_cell_slice_batch' "$@"
