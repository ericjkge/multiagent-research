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


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # never break the agent over a malformed hook payload

    tool = event.get("tool_name", "")
    tool_input = event.get("tool_input") or {}

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
