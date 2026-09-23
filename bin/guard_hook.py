#!/usr/bin/env python3
"""PreToolUse hook enforcing the arena's non-negotiable rules.

Permission modes can be relaxed for autonomy; these rules cannot be, because
breaking any one of them silently invalidates a cell's numbers:

  * training must go through ``arena-train`` (one GPU, metered budget),
  * ``prepare.py`` is read-only (it holds the ground-truth evaluation),
  * no new dependencies,
  * nothing leaves the box.

Denials are returned to the agent as feedback, so it corrects itself rather
than failing the round.
"""

from __future__ import annotations

import json
import os
import re
import sys

DIRECT_TRAIN = re.compile(
    r"(?:^|[|;&]|\s)(?:uv\s+run(?:\s+--\S+)*|python3?|torchrun|accelerate\s+launch)"
    r"\s+[^|;&]*train\.py",
)
ADD_DEPS = re.compile(r"(?:pip\s+install|uv\s+add|uv\s+pip\s+install|conda\s+install|poetry\s+add)")
EXFIL = re.compile(r"(?:^|[|;&]|\s)(?:git\s+push|gh\s+(?:pr|repo|release)|scp|rsync\s+[^|;&]*::)")
ARENA_CMD = re.compile(r"\barena-(?:train|log)\b")

RULES = (
    (
        DIRECT_TRAIN,
        "Training must be launched with `arena-train` (no arguments), not by "
        "invoking train.py directly. arena-train holds the GPU lock and meters "
        "the shared compute budget; bypassing it corrupts every concurrent run.",
    ),
    (
        ADD_DEPS,
        "Adding dependencies is out of scope for this experiment. Use only what "
        "is already in pyproject.toml.",
    ),
    (
        EXFIL,
        "This run is local to the experiment box. Do not push or copy anything "
        "off it.",
    ),
)

PROTECTED_FILES = ("prepare.py",)

# An actual `arena-train` launch (not a mention of it inside ps/grep/cat/tail/until loops).
TRAIN_LAUNCH = re.compile(r"(?:^|[|;&]\s*|\bnohup\s+|\btime\s+)\s*arena-train(?:\s|$)")
# Long enough for a full GPU queue: six agents x ~6 min each, with margin. Claude Code only
# honours this if BASH_MAX_TIMEOUT_MS in the agent environment is at least as large.
TRAIN_TIMEOUT_MS = 3600000


def pin_train_foreground(tool_input: dict) -> None:
    """Keep an arena-train call in the foreground until it finishes.

    Claude Code's default 10-minute shell limit moves a still-running command to the
    background and tells the agent it will be notified -- but in `claude -p` there is no
    later turn to notify, and a session that ends to "wait" kills the training run with
    it. Pin the call's timeout to outlast the whole GPU queue and refuse backgrounding.
    Added 2026-09-22 after sonnet_6 lost ~half its attempts this way.
    """
    updated = dict(tool_input)
    updated["timeout"] = TRAIN_TIMEOUT_MS
    updated.pop("run_in_background", None)
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": "arena-train runs in the foreground with a 60 min timeout.",
            "updatedInput": updated,
        }
    }
    print(json.dumps(payload))
    sys.exit(0)


def deny(reason: str) -> None:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
        # Older hook protocol; harmless when unused.
        "decision": "block",
        "reason": reason,
    }
    print(json.dumps(payload))
    sys.exit(0)


ISOLATION_TOOLS = {"Bash", "Read", "Grep", "Glob", "LS", "Edit", "Write", "MultiEdit"}


def isolation_check(tool: str, tool_input: dict) -> None:
    """Independent cells (ARENA_SHARE_LOG=0): an agent may not touch its peers' work.

    Added Sep 23 after the audit of the Sep 22 cells: every "independent" agent had read the
    shared score table and most had inspected or checked out a peer's commit. The agent's
    private clone holds no peer commit, so this denylist covers the remaining routes: the
    shared table and logs, the other agents' checkouts, the main repository and any git
    command that could pull objects or refs from it.
    """
    if tool not in ISOLATION_TOOLS:
        return
    me = os.environ.get("ARENA_AGENT", "")
    text = " ".join(str(v) for v in tool_input.values())
    run_dir = os.environ.get("ARENA_RUN_DIR", "")
    # Every protocol: no other cell's run directory, no archived cell, no other session's files.
    if run_dir:
        own = os.path.basename(run_dir.rstrip("/"))
        for m in re.finditer(r"runs/([A-Za-z0-9_]+-\d{8}-\d{6})", text):
            if m.group(1) != own:
                deny("Another cell's run directory is off limits.")
        if re.search(r"(?<![\w-])results/", text):
            deny("Archived cells are off limits.")
        if re.search(r"\.claude/projects", text):
            deny("Other sessions' transcripts are off limits.")
    if os.environ.get("ARENA_SHARE_LOG") != "0":
        return
    # Independent cells: inside the run directory only the agent's own files.
    if run_dir:
        own = os.path.basename(run_dir.rstrip("/"))
        allowed = re.compile(rf"^(?:work/open_{me}(?:/|$)|log_{me}\.md|results_{me}\.tsv|slots/{me}(?:/|$)|"
                             rf"runs/{me}(?:/|$)|transcripts/open_{me}_|$)")
        for m in re.finditer(re.escape(own) + r"/([^\s\"'|;&)>]*)", text):
            if not allowed.match(m.group(1)):
                deny("This cell runs agents independently: inside the run directory only your own files "
                     f"(work/open_{me}, log_{me}.md, results_{me}.tsv, slots/{me}, runs/{me}) are yours to read.")
        if re.search(r"(?:^|[\s\"'=])\.\.(?:/|$|\s)", text) or "/../" in text:
            deny("This cell runs agents independently: no paths above your own checkout.")
    if re.search(r"(?:^|[|;&\s])(?:ps|pgrep|pstree|top|htop|lsof|fuser|w|who)\b", text) or "/proc/" in text:
        deny("This cell runs agents independently: process lists show other agents' commands; "
             "arena-train runs in the foreground and returns when your run is done.")
    for m in re.finditer(r"open-(a\d+)", text):   # /tmp/claude-*/...-work-open-aN task outputs
        if m.group(1) != me:
            deny("This cell runs agents independently: another agent's session files are off limits.")
    repo = os.path.dirname(os.environ.get("ARENA_UV_ENV", "") or "")
    checks = [
        (re.compile(r"results\.tsv"), "the shared score table"),
        (re.compile(r"results_(a\d+)\.tsv"), "another agent's score table"),
        (re.compile(r"(?<![\w-])log\.md\b"), "the shared log"),
        (re.compile(r"(?<![\w-])log_(a\d+)\.md"), "another agent's log"),
        (re.compile(r"work/open_(a\d+)"), "another agent's checkout"),
        (re.compile(r"refs/arena|--all\b|\breflog\b|for-each-ref|lost-found|\bfsck\b|cat-file"),
         "git history outside your own commits"),
        (re.compile(r"\bgit\s+(?:fetch|pull|remote|clone|submodule|worktree)\b"), "fetching from another repository"),
    ]
    for pat, what in checks:
        for m in pat.finditer(text):
            peer = m.group(1) if m.groups() else None
            if peer is not None and peer == me:
                continue
            deny(f"This cell runs agents independently: {what} is off limits.")
    if repo and repo in text and f"{repo}/.venv" not in text:
        deny("This cell runs agents independently: the main repository is off limits; work in your own checkout.")
    if run_dir:
        top = os.path.dirname(os.path.dirname(run_dir.rstrip("/")))   # the orchestrator repo
        for m in re.finditer(re.escape(top) + r"/([^\s\"'|;&)>]*)", text):
            if not m.group(1).startswith("runs/" + os.path.basename(run_dir.rstrip("/"))):
                deny("This cell runs agents independently: the orchestrator repository is off limits except your own run directory.")


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # never break the agent over a malformed hook payload

    tool = event.get("tool_name", "")
    tool_input = event.get("tool_input") or {}
    isolation_check(tool, tool_input)

    if tool in {"Bash", "BashOutput"}:
        command = tool_input.get("command", "")
        # `arena-train` and `arena-log` are the sanctioned entry points, so the
        # bare-command patterns must not fire on them -- but a disallowed
        # command chained onto one still has to be caught.
        if not ARENA_CMD.search(command):
            for pattern, reason in RULES:
                if pattern.search(command):
                    deny(reason)
        elif ADD_DEPS.search(command) or EXFIL.search(command):
            deny("Disallowed command chained onto an arena command.")
        elif tool == "Bash" and TRAIN_LAUNCH.search(command):
            pin_train_foreground(tool_input)

    if tool in {"Edit", "Write", "NotebookEdit", "MultiEdit"}:
        path = str(tool_input.get("file_path", ""))
        if any(path.endswith(p) for p in PROTECTED_FILES):
            deny(
                f"{path} is read-only: it defines the ground-truth evaluation. "
                "Only train.py may be modified."
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
