"""Cell configuration.

A *cell* is one square of the experiment grid: a model family crossed with an
agent count (e.g. ``sonnet_3``).  Everything that distinguishes one cell from
another lives in a YAML file under ``configs/``; the orchestrator itself is
identical across cells.

Compute matching is the point of this file.  Every cell is given the same
budget of **training runs**, and a run is a run: a crash spends a slot exactly
like a success does.  The cell is allotted N attempts at the GPU and wasting
one on a bug is a real cost to the group, which is precisely what the
error-propagation arm of the experiment is trying to measure.

A consequence worth stating: ``max_rounds`` is an upper bound, not a plan.  A
cell that crashes a lot completes fewer rounds on the same compute.  That is
the honest behaviour -- the alternative, giving crashes back their compute,
would hide the cost of a failure cascade.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 300s of training plus startup, compile and eval overhead.  Used only to turn
# the human-facing "36 runs" budget into the seconds the arena actually meters.
DEFAULT_NOMINAL_RUN_SECONDS = 330.0

KNOWN_HARNESSES = {"claude_code", "opencode", "fake"}


@dataclass(frozen=True)
class AgentSpec:
    """One researcher in the arena."""

    id: str
    model: str
    effort: str = "medium"
    harness: str = "claude_code"

    def __post_init__(self) -> None:
        if self.effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"agent {self.id}: bad effort {self.effort!r}")
        if self.harness not in KNOWN_HARNESSES:
            raise ValueError(f"agent {self.id}: unknown harness {self.harness!r}")


@dataclass
class CellConfig:
    cell_id: str
    agents: list[AgentSpec]

    # Compute matching -------------------------------------------------
    bon: int = 1
    train_run_budget: int = 36
    nominal_run_seconds: float = DEFAULT_NOMINAL_RUN_SECONDS

    # Training mechanics -----------------------------------------------
    train_timeout_s: int = 600
    max_train_attempts: int = 3

    # Substrate ---------------------------------------------------------
    autoresearch_repo: str = "~/autoresearch"
    autoresearch_commit: str = ""

    # Cost guards --------------------------------------------------------
    max_budget_usd_per_session: float = 5.0
    cell_budget_usd: float = 100.0

    # Protocol knobs -----------------------------------------------------
    # "rounds": propose -> select -> respond in lockstep, best candidate of the
    #           round becomes everyone's baseline (the arena as first built).
    # "open":   no rounds. Each agent works asynchronously in one long session
    #           against its own share of the run budget, declares a distinct
    #           approach in a slot, publishes findings and disconfirmations to
    #           the shared log, and adopts a peer's approach only after a
    #           clearly better measured result -- the protocol of Park et al.,
    #           "Scaling Discovery through Test-Time Communication"
    #           (arXiv 2609.21032).  With open_share_log: false the same agents
    #           run blind to each other, which is that paper's independent
    #           (best@k) control at matched compute.
    protocol: str = "rounds"
    solo_self_critique: bool = False
    phase_timeout_s: dict[str, int] = field(
        default_factory=lambda: {"propose": 900, "select": 3600, "respond": 1200}
    )
    # Open-protocol knobs (ignored under "rounds").
    open_share_log: bool = True          # False: each agent sees only its own entries
    open_per_agent_quota: bool = True    # each agent gets budget // n_agents runs
    open_max_resumes: int = 8            # a session that stops early is resumed this many times
    open_session_timeout_s: int = 5400   # wall clock per session segment

    # Execution ----------------------------------------------------------
    fake_gpu: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.agents:
            raise ValueError("a cell needs at least one agent")
        ids = [a.id for a in self.agents]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate agent ids: {ids}")
        if self.bon < 1:
            raise ValueError("bon must be >= 1")
        if self.train_run_budget < 1:
            raise ValueError("train_run_budget must be >= 1")
        if self.protocol not in {"rounds", "open"}:
            raise ValueError(f"protocol must be 'rounds' or 'open', not {self.protocol!r}")
        if self.protocol == "open" and self.bon != 1:
            raise ValueError("the open protocol has no best-of-N: agents sample as they like; set bon: 1")

    @property
    def per_agent_quota(self) -> int:
        """Runs each agent may claim under the open protocol (0 = unlimited)."""
        if self.protocol != "open" or not self.open_per_agent_quota:
            return 0
        return max(1, self.train_run_budget // self.n_agents)

    # -- derived ----------------------------------------------------------

    @property
    def n_agents(self) -> int:
        return len(self.agents)

    @property
    def runs_per_round(self) -> int:
        """Candidates trained per round: one per agent per best-of-N sample."""
        return self.n_agents * self.bon

    @property
    def max_rounds(self) -> int:
        """Round cap if nothing crashes.

        An upper bound only: the run budget is the real constraint, and every
        retry after a crash brings the last round forward.
        """
        return max(1, self.train_run_budget // self.runs_per_round)

    @property
    def nominal_gpu_hours(self) -> float:
        """For the runbook and the invoice; nothing is enforced in seconds."""
        return self.train_run_budget * self.nominal_run_seconds / 3600

    @property
    def respond_enabled(self) -> bool:
        """A solo agent has nobody to respond to."""
        return self.n_agents > 1 or self.solo_self_critique

    @property
    def repo_path(self) -> Path:
        return Path(self.autoresearch_repo).expanduser().resolve()

    def fingerprint(self) -> str:
        """Stable hash of the config, recorded with every run for provenance."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["agents"] = [asdict(a) for a in self.agents]
        return d

    def summary(self) -> str:
        if self.protocol == "open":
            share = "shared log" if self.open_share_log else "NO shared log (independent)"
            quota = f"{self.per_agent_quota} runs each" if self.per_agent_quota else "first come first served"
            return (
                f"{self.cell_id}: open protocol, {self.n_agents} agent(s), {share}, "
                f"{self.train_run_budget} runs ~ {self.nominal_gpu_hours:.2f} GPU-h ({quota})"
            )
        return (
            f"{self.cell_id}: {self.n_agents} agent(s) x BoN={self.bon} "
            f"= {self.runs_per_round} candidates/round, "
            f"<={self.max_rounds} rounds, "
            f"{self.train_run_budget} runs ~ {self.nominal_gpu_hours:.2f} GPU-h"
        )


def load_cell(path: str | Path) -> CellConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping")

    agents_raw = raw.pop("agents", [])
    default_harness = raw.pop("harness", "claude_code")
    agents = [
        AgentSpec(
            id=a["id"],
            model=a["model"],
            effort=a.get("effort", "medium"),
            harness=a.get("harness", default_harness),
        )
        for a in agents_raw
    ]

    unknown = set(raw) - {f.name for f in CellConfig.__dataclass_fields__.values()}
    if unknown:
        raise ValueError(f"{path}: unknown config keys: {sorted(unknown)}")

    return CellConfig(agents=agents, **raw)
