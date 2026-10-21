"""Logging: console + per-run file (UTC timestamps) and a transformation log.

`TransformationLog` implements the rule "every transformation must be logged":
each data-changing step records rows before/after and its parameters as one
JSON line in <run_dir>/transformations.jsonl.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOGGER_NAME = "solar_flare"


def get_logger(child: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{child}" if child else LOGGER_NAME)


def setup_logging(run_dir, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level))
    logger.propagate = False
    for h in list(logger.handlers):          # idempotent: no duplicate handlers
        logger.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)sZ | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S")
    fmt.converter = time.gmtime               # UTC
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    file_handler = logging.FileHandler(Path(run_dir) / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


class TransformationLog:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / "transformations.jsonl"

    def record(self, stage: str, action: str, rows_before: Optional[int] = None,
               rows_after: Optional[int] = None, **details) -> dict:
        entry = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "action": action,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": (rows_before - rows_after
                             if rows_before is not None and rows_after is not None else None),
            "details": details,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        get_logger("transform").info("%s :: %s (rows %s -> %s)", stage, action, rows_before, rows_after)
        return entry

    def read(self) -> list:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
