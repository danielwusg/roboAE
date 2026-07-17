from __future__ import annotations

import json
import sys

from relay import relay_entry_main


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("relay entry expects one specification")
    return relay_entry_main(json.loads(sys.argv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
