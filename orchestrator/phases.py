"""The three phases of a round: propose, respond, finalize.

Propose and respond are tool-free calls constrained by a JSON schema, so the
agents' output is data rather than prose to be parsed.  Finalize is a real
Claude Code session with tools, because the experiment wants agents that debug
their own crashes -- error propagation is one of the things under study, and it
cannot be observed if the orchestrator does the repairs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AgentSpec, CellConfig
from .harness import Harness, HarnessResult, new_session_id
from .sharedlog import SharedLog

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
BIN = Path(__file__).resolve().parent.parent / "bin"

# Tools the finalize session needs: edit the file, run the metered trainer,
# read the log and the traceback.
FINALIZE_TOOLS = ("Bash", "Read", "Edit", "Write", "Grep", "Glob")

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


def agent_env(ctx: PhaseContext, agent: AgentSpec, variant: int) -> dict[str, str]:
    return {
        "PATH": f"{BIN}:{Path('/usr/bin')}:{Path('/bin')}:" + __import__("os").environ.get("PATH", ""),
        "ARENA_RUN_DIR": str(ctx.run_dir),
        "ARENA_AGENT": agent.id,
        "ARENA_VARIANT": str(variant),
        "ARENA_ROUND": str(ctx.round_idx),
        "ARENA_TIMEOUT_S": str(ctx.cfg.train_timeout_s),
        "ARENA_FAKE_GPU": "1" if ctx.cfg.fake_gpu else "0",
        "ARENA_SEED": str(ctx.cfg.seed),
        "ARENA_UV_ENV": str(ctx.cfg.repo_path / ".venv"),
        "ARENA_FAKE_BASE_BPB": str(getattr(ctx, "baseline_bpb", 0.9979)),
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
    )
    return harness.query(
        prompt,
        cwd=ctx.run_dir,
        model=agent.model,
        effort=agent.effort,
        schema=RESPOND_SCHEMA,
        tools=(),
        resume=session_id,
        timeout_s=ctx.cfg.phase_timeout_s["respond"],
        append_system_prompt=system_prompt(ctx, agent),
        max_budget_usd=ctx.cfg.max_budget_usd_per_session,
        transcript_path=ctx.run_dir / "transcripts" / f"r{ctx.round_idx:02d}_{agent.id}_respond.json",
    )


def run_finalize(
    harness: Harness,
    agent: AgentSpec,
    ctx: PhaseContext,
    session_id: str | None,
    worktree: Path,
    variant: int,
) -> HarnessResult:
    prompt = render(
        "finalize.md",
        ROUND=str(ctx.round_idx),
        WORKTREE=str(worktree),
        MAX_ATTEMPTS=str(ctx.cfg.max_train_attempts),
    )
    return harness.query(
        prompt,
        cwd=worktree,
        model=agent.model,
        effort=agent.effort,
        schema=None,
        tools=FINALIZE_TOOLS,
        resume=session_id,
        fork=bool(session_id),
        session_id=None if session_id else new_session_id(),
        permission_mode="bypassPermissions",
        timeout_s=ctx.cfg.phase_timeout_s["finalize"],
        append_system_prompt=system_prompt(ctx, agent),
        settings=ctx.settings_path,
        add_dirs=[ctx.run_dir],
        env=agent_env(ctx, agent, variant),
        max_budget_usd=ctx.cfg.max_budget_usd_per_session,
        transcript_path=ctx.run_dir / "transcripts"
        / f"r{ctx.round_idx:02d}_{agent.id}_v{variant}_finalize.json",
    )


def read_candidate_json(worktree: Path) -> dict:
    path = Path(worktree) / "candidate.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}
