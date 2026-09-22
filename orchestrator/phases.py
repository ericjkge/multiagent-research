"""The three phases of a round: propose, select, respond.

**Propose** is a tool-free call constrained by a JSON schema, so the output is
data rather than prose to be parsed, and so an agent cannot peek at the repo
or the other agents while it is supposed to be thinking independently.

**Select** is a real Claude Code session with tools: the agent claims a run
slot, implements its idea and debugs its own crashes.  The orchestrator
deliberately does not do the repairs -- error propagation is one of the things
under study and it cannot be observed if the harness quietly fixes things.

**Respond** comes *after* the measurement.  It keeps the schema, for a
canonical message, but also gets tools, because the experiment gives agents
write access to the shared log: they can read the run logs, dig into a peer's
crash, and post threaded replies with ``arena-log``.  ``arena-train`` refuses
to run in this phase, so the extra reach cannot be used to buy extra compute.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AgentSpec, CellConfig
from .harness import Harness, HarnessResult, new_session_id
from .sharedlog import SharedLog

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
BIN = Path(__file__).resolve().parent.parent / "bin"

# Tools the select session needs: edit the file, claim a slot and run the
# metered trainer, read the log and the traceback.
SELECT_TOOLS = ("Bash", "Read", "Edit", "Write", "Grep", "Glob")

# Respond needs to read (run logs, results, a peer's train.py) and to run
# `arena-log`, but has no reason to edit anything.  arena-train blocks itself
# in this phase.
RESPOND_TOOLS = ("Bash", "Read", "Grep", "Glob")

PROPOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 120},
        "justification": {"type": "string", "maxLength": 400},
        "detail": {"type": "string", "maxLength": 3000},
        "notes": {"type": "string", "maxLength": 1000},
    },
    "required": ["title", "justification", "detail"],
    "additionalProperties": False,
}

RESPOND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "response": {"type": "string", "maxLength": 3000},
        "notes": {"type": "string", "maxLength": 1000},
    },
    "required": ["response"],
    "additionalProperties": False,
}


def render(template: str, **vars: str) -> str:
    """``{{TOKEN}}`` substitution.

    Deliberately not ``str.format``: the templates carry Python source, which is
    full of braces.
    """
    text = (PROMPTS / template).read_text()
    for key, value in vars.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


@dataclass
class PhaseContext:
    cfg: CellConfig
    round_idx: int
    log: SharedLog
    run_dir: Path
    train_py: str
    diff_vs_origin: str
    settings_path: Path
    baseline_bpb: float = 0.0  # consumed by the fake-GPU stub


def system_prompt(ctx: PhaseContext, agent: AgentSpec) -> str:
    peers = [a.id for a in ctx.cfg.agents if a.id != agent.id]
    return render(
        "system_agent.md",
        AGENT_ID=agent.id,
        N_AGENTS=str(ctx.cfg.n_agents),
        PEER_IDS=", ".join(peers) if peers else "none — you are working alone this cell",
        LOG_PATH=str(ctx.log.markdown),
        RESULTS_PATH=str(ctx.log.results_tsv),
        SCRATCHPAD_PATH=str(ctx.log.scratchpad),
    )


def write_agent_settings(run_dir: Path) -> Path:
    """Settings that carry the rule-enforcement hook into every agent session."""
    path = run_dir / "agent_settings.json"
    path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "*",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": str(BIN / "guard_hook.py"),
                                }
                            ],
                        }
                    ]
                }
            },
            indent=2,
        )
    )
    return path


def agent_env(
    ctx: PhaseContext, agent: AgentSpec, variant: int, phase: str = "select"
) -> dict[str, str]:
    return {
        # bin/ first so `arena-train` and `arena-log` resolve to ours.
        "PATH": os.pathsep.join([str(BIN), "/usr/bin", "/bin", os.environ.get("PATH", "")]),
        "ARENA_RUN_DIR": str(ctx.run_dir),
        "ARENA_AGENT": agent.id,
        "ARENA_VARIANT": str(variant),
        "ARENA_ROUND": str(ctx.round_idx),
        # arena-train refuses to claim a slot outside the select phase.
        "ARENA_PHASE": phase,
        "ARENA_TIMEOUT_S": str(ctx.cfg.train_timeout_s),
        "ARENA_FAKE_GPU": "1" if ctx.cfg.fake_gpu else "0",
        "ARENA_SEED": str(ctx.cfg.seed),
        "ARENA_UV_ENV": str(ctx.cfg.repo_path / ".venv"),
        "ARENA_FAKE_BASE_BPB": str(ctx.baseline_bpb or 0.9979),
        # A training run is ~6 minutes; Claude Code's Bash tool kills commands at 2 minutes by
        # default, which killed arena-train mid-run and lost the slot. 15 minutes, both limits.
        "BASH_DEFAULT_TIMEOUT_MS": "900000",
        "BASH_MAX_TIMEOUT_MS": "900000",
    }


# -- phase calls ---------------------------------------------------------


def run_propose(harness: Harness, agent: AgentSpec, ctx: PhaseContext) -> HarnessResult:
    diff_section = ""
    if ctx.diff_vs_origin.strip():
        diff_section = (
            "This is how the baseline has drifted from the original code so far:\n\n"
            "<diff_vs_original>\n" + ctx.diff_vs_origin + "\n</diff_vs_original>\n"
        )

    prompt = render(
        "propose.md",
        ROUND=str(ctx.round_idx),
        LOG_MD=ctx.log.markdown.read_text(),
        TRAIN_PY=ctx.train_py,
        DIFF_SECTION=diff_section,
    )
    return harness.query(
        prompt,
        cwd=ctx.run_dir,
        model=agent.model,
        effort=agent.effort,
        schema=PROPOSE_SCHEMA,
        tools=(),
        session_id=new_session_id(),
        phase="propose",
        timeout_s=ctx.cfg.phase_timeout_s["propose"],
        append_system_prompt=system_prompt(ctx, agent),
        max_budget_usd=ctx.cfg.max_budget_usd_per_session,
        transcript_path=ctx.run_dir / "transcripts" / f"r{ctx.round_idx:02d}_{agent.id}_propose.json",
    )


def run_respond(
    harness: Harness,
    agent: AgentSpec,
    ctx: PhaseContext,
    session_id: str,
    own_title: str,
) -> HarnessResult:
    prompt = render(
        "respond.md",
        ROUND=str(ctx.round_idx),
        LOG_MD=ctx.log.markdown.read_text(),
        OWN_TITLE=own_title,
        RESULTS_PATH=str(ctx.log.results_tsv),
    )
    return harness.query(
        prompt,
        cwd=ctx.run_dir,
        model=agent.model,
        effort=agent.effort,
        schema=RESPOND_SCHEMA,
        tools=RESPOND_TOOLS,
        resume=session_id,
        # Forked: respond resumes the select session, which best-of-N already
        # forked.  Resuming it in place would make the next round's history
        # depend on which variant happened to be picked here.
        fork=bool(session_id),
        session_id=None if session_id else new_session_id(),
        permission_mode="bypassPermissions",
        phase="respond",
        timeout_s=ctx.cfg.phase_timeout_s["respond"],
        append_system_prompt=system_prompt(ctx, agent),
        settings=ctx.settings_path,
        add_dirs=[ctx.run_dir],
        env=agent_env(ctx, agent, variant=0, phase="respond"),
        max_budget_usd=ctx.cfg.max_budget_usd_per_session,
        transcript_path=ctx.run_dir / "transcripts" / f"r{ctx.round_idx:02d}_{agent.id}_respond.json",
    )


def run_select(
    harness: Harness,
    agent: AgentSpec,
    ctx: PhaseContext,
    session_id: str | None,
    worktree: Path,
    variant: int,
    runs_left: int,
) -> HarnessResult:
    prompt = render(
        "select.md",
        ROUND=str(ctx.round_idx),
        WORKTREE=str(worktree),
        MAX_ATTEMPTS=str(ctx.cfg.max_train_attempts),
        RUNS_LEFT=str(runs_left),
        LOG_MD=ctx.log.markdown.read_text(),
    )
    return harness.query(
        prompt,
        cwd=worktree,
        model=agent.model,
        effort=agent.effort,
        schema=None,
        tools=SELECT_TOOLS,
        resume=session_id,
        fork=bool(session_id),
        session_id=None if session_id else new_session_id(),
        permission_mode="bypassPermissions",
        phase="select",
        timeout_s=ctx.cfg.phase_timeout_s["select"],
        append_system_prompt=system_prompt(ctx, agent),
        settings=ctx.settings_path,
        add_dirs=[ctx.run_dir],
        env=agent_env(ctx, agent, variant, phase="select"),
        max_budget_usd=ctx.cfg.max_budget_usd_per_session,
        transcript_path=ctx.run_dir / "transcripts"
        / f"r{ctx.round_idx:02d}_{agent.id}_v{variant}_select.json",
    )


_PLACEHOLDER_WORDS = {
    "placeholder", "tbd", "todo", "n/a", "na", "none", "example",
    "test", "idea", "untitled", "...", "",
}
MIN_DETAIL_CHARS = 40


def is_degenerate_proposal(data: dict) -> str:
    """Why this proposal is contentless, or "" if it is fine.

    A schema-validated response can still say nothing -- a small model at low
    effort will sometimes return ``{"title": "Placeholder", ...}``.  It passes
    the schema, so nothing downstream notices, and it silently becomes a data
    point: it drags the round's measured idea-diversity to zero and counts as
    an agent that participated.  Catch it at the door and mark it.
    """
    title = (data.get("title") or "").strip()
    justification = (data.get("justification") or "").strip()
    detail = (data.get("detail") or "").strip()

    if not title:
        return "no title"
    if title.strip(" .").lower() in _PLACEHOLDER_WORDS:
        return f"placeholder title ({title!r})"
    if justification.strip(" .").lower() in _PLACEHOLDER_WORDS:
        return f"placeholder justification ({justification!r})"
    if len(detail) < MIN_DETAIL_CHARS:
        return f"detail is {len(detail)} chars, too short to implement"
    return ""


def read_candidate_json(worktree: Path) -> dict:
    path = Path(worktree) / "candidate.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
