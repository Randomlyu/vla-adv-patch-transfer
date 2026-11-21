"""Structured per-episode metrics for LIBERO evaluation (P0).

Contract
--------
* ``metrics.jsonl``: exactly one JSON object per finished episode, UTF-8, one line each.
* ``summary.json``: one aggregate object written once when the run ends.

All downstream consumers (tables, CSV export, the analysis UI) read these two files
instead of parsing stdout or the ``.log`` transcript.

The writer is deliberately dependency-free so it can be imported before torch / LIBERO
are available (useful for unit checks on a laptop).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Field order is fixed: it makes the JSONL diff-friendly and keeps the CSV header stable.
EPISODE_FIELDS = (
    "run_id",
    "episode",
    "task_suite",
    "task_id",
    "task_description",
    "init_state_idx",
    "mode",              # "clean" | "attacked"
    "success",
    "steps",             # env steps executed after the num_steps_wait warm-up
    "max_steps",
    "exception",         # "TypeName: message" or None
    "invalid",           # True if the episode raised; success is meaningless then
    "timestamp",
    "patch_path",        # absolute path or None for clean runs
    "checkpoint",
    "seed",
    "x",
    "y",
    "angle",
    "shx",
    "shy",
    "video_path",
    "wall_time_sec",
)


class MetricsWriter:
    """Append-only JSONL writer with a fixed schema."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self.records: List[Dict[str, Any]] = []

    def write_episode(self, record: Dict[str, Any]) -> Dict[str, Any]:
        missing = [key for key in EPISODE_FIELDS if key not in record]
        if missing:
            raise KeyError(f"metrics record is missing required fields: {missing}")
        row = {key: record[key] for key in EPISODE_FIELDS}
        if not row["timestamp"]:
            row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.records.append(row)
        return row

    def close(self, summary: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        if not self._fh.closed:
            self._fh.close()
        if summary is None:
            return None
        summary_path = self.path.parent / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return summary_path


def load_metrics(path: str | os.PathLike) -> List[Dict[str, Any]]:
    """Read ``metrics.jsonl`` back into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Minimal aggregate used to fill ``summary.json``."""
    rows = list(rows)
    valid = [row for row in rows if not row["invalid"]]
    episodes = len(rows)
    invalid = sum(1 for row in rows if row["invalid"])
    successes = sum(1 for row in valid if row["success"])
    return {
        "episodes": episodes,
        "valid_episodes": len(valid),
        "invalid_episodes": invalid,
        "successes": successes,
        "success_rate": (successes / len(valid)) if valid else None,
    }
