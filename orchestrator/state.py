"""Checkpointing.

A cell is three hours of GPU time and real money.  If the orchestrator dies in
round 5 it must resume in round 5, not start over.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunState:
    cell_id: str
    tag: str
    origin_commit: str
    baseline_commit: str
    baseline_bpb: float
    round_idx: int = 0
    completed_rounds: list[int] = field(default_factory=list)
    total_cost_usd: float = 0.0
    started_at: float = 0.0
    finished_at: float | None = None
    stop_reason: str = ""

    @classmethod
    def path_for(cls, run_dir: Path) -> Path:
        return Path(run_dir) / "state.json"

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        data = json.loads(cls.path_for(run_dir).read_text())
        return cls(**data)

    def save(self, run_dir: Path) -> None:
        tmp = self.path_for(run_dir).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(self.path_for(run_dir))
