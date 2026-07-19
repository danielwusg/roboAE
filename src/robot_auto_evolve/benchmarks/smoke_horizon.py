from __future__ import annotations

import os


def smoke_horizon_override() -> int | None:
    """Test-only: set by --smoke-horizon (threaded via SimulatorProcess's env overlay).
    When active, simulator workers skip their strict episode.horizon==catalog check so a
    smoke can run a shorter horizon (episode.horizon is already capped by _smoke_plan).
    Returns the cap, or None when not active."""
    value = os.environ.get("ROBOT_AE_SMOKE_HORIZON")
    if not value:
        return None
    cap = int(value)
    return cap if cap > 0 else None
