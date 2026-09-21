"""Shared loading helpers for the analysis scripts."""

from __future__ import annotations

import json
from pathlib import Path


def load_records(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "log.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_json(run_dir: Path, name: str) -> dict:
    path = Path(run_dir) / name
    return json.loads(path.read_text()) if path.exists() else {}


def load_timeline(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "gpu_timeline.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def of_type(records: list[dict], kind: str) -> list[dict]:
    return [r for r in records if r.get("t") == kind]
