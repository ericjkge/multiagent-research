"""Reading the ground truth out of a training run.

``arena-train`` owns the GPU and writes ``arena_run.json``; ``train.py`` writes
``run.log``.  Between them they are the authoritative record of what happened,
which is why the orchestrator parses them directly instead of trusting an
agent's own account of its result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

METRIC = re.compile(r"^(?P<key>[a-z_]+):\s+(?P<value>-?[\d.]+)\s*$", re.MULTILINE)

# Keys train.py prints in its final summary block.
NUMERIC_KEYS = (
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
)


@dataclass
class RunMetrics:
    ok: bool
    val_bpb: float | None = None
    peak_vram_mb: float | None = None
    training_seconds: float | None = None
    total_seconds: float | None = None
    extra: dict | None = None
    error: str = ""

    @property
    def status(self) -> str:
        return "ok" if self.ok else "crash"


def parse_run_log(path: Path) -> RunMetrics:
    """A run counts only if it printed a val_bpb; anything else is a crash."""
    path = Path(path)
    if not path.exists():
        return RunMetrics(ok=False, error="no run.log was produced")

    text = path.read_text(errors="replace")
    found = {
        m.group("key"): float(m.group("value"))
        for m in METRIC.finditer(text)
        if m.group("key") in NUMERIC_KEYS
    }

    if "val_bpb" not in found:
        tail = "\n".join(text.splitlines()[-25:])
        return RunMetrics(ok=False, error=tail.strip() or "empty run.log")

    return RunMetrics(
        ok=True,
        val_bpb=found["val_bpb"],
        peak_vram_mb=found.get("peak_vram_mb"),
        training_seconds=found.get("training_seconds"),
        total_seconds=found.get("total_seconds"),
        extra={k: v for k, v in found.items() if k not in
               {"val_bpb", "peak_vram_mb", "training_seconds", "total_seconds"}},
    )


def read_arena_run(workdir: Path) -> dict | None:
    """The GPU-lock record for the last run in this worktree, if any."""
    path = Path(workdir) / "arena_run.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def assert_no_overlap(timeline_path: Path, slack_s: float = 1.0) -> list[tuple[dict, dict]]:
    """Proof that the single-GPU assumption held.

    Returns offending pairs; an empty list means every training run in the cell
    was serialized and its timings are comparable.
    """
    path = Path(timeline_path)
    if not path.exists():
        return []
    runs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    runs.sort(key=lambda r: r["started_at"])
    overlaps = []
    for earlier, later in zip(runs, runs[1:]):
        if later["started_at"] < earlier["ended_at"] - slack_s:
            overlaps.append((earlier, later))
    return overlaps
