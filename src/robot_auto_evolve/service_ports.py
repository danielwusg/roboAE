from __future__ import annotations


MIN_STATIC_SERVICE_PORT = 1024
MAX_STATIC_SERVICE_PORT = 32767


def checked_static_service_port(value: int) -> int:
    if type(value) is not int or not MIN_STATIC_SERVICE_PORT <= value <= MAX_STATIC_SERVICE_PORT:
        raise ValueError(
            f"service port must stay in {MIN_STATIC_SERVICE_PORT}-{MAX_STATIC_SERVICE_PORT}; "
            "higher ports overlap Linux ephemeral clients"
        )
    return value
