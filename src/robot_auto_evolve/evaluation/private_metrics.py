from __future__ import annotations

import math
import re
from typing import Any, Mapping

from robot_auto_evolve.protocol import StrictSchemaError


MAX_PRIVATE_METRICS = 16
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def validate_private_metrics(value: Any, path: str = "private_metrics") -> dict[str, bool | float]:
    if not isinstance(value, Mapping) or not 1 <= len(value) <= MAX_PRIVATE_METRICS:
        raise StrictSchemaError(f"{path}: expected 1 to {MAX_PRIVATE_METRICS} metrics")
    result: dict[str, bool | float] = {}
    for name, metric in sorted(value.items()):
        if type(name) is not str or _NAME.fullmatch(name) is None:
            raise StrictSchemaError(f"{path}: invalid metric name")
        if type(metric) is bool:
            result[name] = metric
        elif type(metric) in (int, float) and math.isfinite(float(metric)):
            result[name] = float(metric)
        else:
            raise StrictSchemaError(f"{path}.{name}: expected finite scalar or bool")
    return result
