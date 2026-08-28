"""Logging setup.

Plain text locally, single-line JSON in production so container logs stay greppable.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Anything passed via logger.info(..., extra={...}) rides along.
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Uvicorn installs its own handlers; defer to ours to avoid duplicate lines.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
