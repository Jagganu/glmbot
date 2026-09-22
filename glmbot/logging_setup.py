"""Centralized logging: Rich console on desktop, plain on minimal installs.

Features
  - ``setup_logging(verbose, log_file)`` — idempotent, safe to call twice.
  - Rotating file handler under ``data/logs/glmbot.log`` (200 KB x 5).
  - Secret redaction filter (masks api keys / tokens in log records).
  - ``get_logger(name)`` convenience wrapper.

Environment
  - ``GLMBOT_LOG_LEVEL`` overrides the default level (DEBUG with -v else INFO).
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import re
from pathlib import Path
from typing import Optional

_SECRET_PATTERNS = [
    re.compile(r"(api[_-]?secret\s*[:=]\s*)(['\"]?)([A-Za-z0-9/+_=.-]{8,})\2", re.IGNORECASE),
    re.compile(r"(bot_token\s*[:=]\s*)(['\"]?)([A-Za-z0-9:_\-]{8,})\2", re.IGNORECASE),
    re.compile(r"(signature=[a-f0-9]{16,})", re.IGNORECASE),
]

_CONFIGURED = False


class _RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            redacted = msg
            for pat in _SECRET_PATTERNS:
                redacted = pat.sub(lambda m: m.group(0)[: len(m.group(1)) + 4] + "****", redacted)
            if redacted != msg:
                record.msg = redacted
                record.args = ()
        except Exception:
            pass
        return True


def _log_file_path() -> Path:
    return Path("data") / "logs" / "glmbot.log"


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure root + glmbot loggers. Safe to call multiple times."""
    global _CONFIGURED
    level_name = os.environ.get("GLMBOT_LOG_LEVEL", "")
    if level_name:
        level = getattr(logging, level_name.upper(), logging.INFO)
    else:
        level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    # Remove our previous handlers so repeated calls (tests/REPL) don't duplicate.
    for h in [h for h in root.handlers if getattr(h, "_glmbot", False)]:
        root.removeHandler(h)

    fmt_plain = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    try:
        from rich.logging import RichHandler

        from .ui import console as _get_console

        console = _get_console()
        handler: logging.Handler = RichHandler(
            console=console, rich_tracebacks=True, show_path=False,
            markup=True, log_time_format="[%H:%M:%S]",
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    except Exception:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt_plain, datefmt=datefmt))

    handler.addFilter(_RedactFilter())
    handler._glmbot = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # File handler (best effort — never crash the bot over logging).
    try:
        path = Path(log_file) if log_file else _log_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            str(path), maxBytes=200_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(fmt_plain, datefmt=datefmt))
        fh.addFilter(_RedactFilter())
        fh._glmbot = True  # type: ignore[attr-defined]
        root.addHandler(fh)
    except Exception:
        pass

    _CONFIGURED = True
    return logging.getLogger("glmbot")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
