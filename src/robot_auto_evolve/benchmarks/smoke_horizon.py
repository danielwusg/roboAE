from __future__ import annotations

import os


def smoke_horizon_override() -> int | None:
    value = os.environ.get("ROBOT_AE_SMOKE_HORIZON")
    if not value:
        return None
    cap = int(value)
    return cap if cap > 0 else None
