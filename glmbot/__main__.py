"""Allow `python -m glmbot` as an alias for `python bot.py`."""

from __future__ import annotations

import sys

if __name__ == "__main__":
    sys.path.insert(0, ".")
    from bot import main

    raise SystemExit(main())
