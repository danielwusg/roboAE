#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec "$ROOT/launch/route.sh" 'openvla_simpler_google_va' "$@"
