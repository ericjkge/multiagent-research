"""Coding-agent harnesses.

``Harness`` is a protocol rather than a class hierarchy so the mixed-model cell
(OpenCode with Sonnet + GPT + Gemini) can be added later as one more
implementation without touching the arena loop.

The Claude Code implementation drives the CLI in headless mode.  Two features
carry most of the weight:

``--json-schema``
    makes propose/respond return validated ``structured_output`` instead of
    prose we would have to regex.

``--resume`` / ``--fork-session``
    let an agent's select step inherit the context of its own proposal, and let
    best-of-N fork that context into N independent samples.

``FakeHarness`` answers every phase from canned data and makes a real edit to
``train.py``, so the whole loop can be exercised for **zero** API spend.
``OpenCodeHarness`` is the seam for the mixed-model cell.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


class HarnessError(RuntimeError):
    pass


@dataclass
class HarnessResult:
    text: str
    structured: dict | None
    session_id: str
    cost_usd: float
    num_turns: int
    is_error: bool
    duration_s: float
    usage: dict = field(default_factory=dict)
    model_usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def models_used(self) -> list[str]:
        return sorted(self.model_usage)


class Harness(Protocol):
    name: str

    def query(self, prompt: str, **kwargs: Any) -> HarnessResult: ...


class ClaudeCodeHarness:
    name = "claude_code"

    def __init__(self, binary: str = "claude", inherit_user_settings: bool = False):
        self.binary = binary
        # Personal settings and CLAUDE.md files would make one researcher's run
        # incomparable to another's, so they are excluded by default.
        self.inherit_user_settings = inherit_user_settings

    def query(
        self,
        prompt: str,
        *,
        cwd: Path,
        model: str,
        effort: str = "medium",
        schema: dict | None = None,
        tools: Sequence[str] | None = (),
        resume: str | None = None,
        fork: bool = False,
        session_id: str | None = None,
        permission_mode: str | None = None,
        max_budget_usd: float | None = None,
        timeout_s: float = 900,
        append_system_prompt: str | None = None,
        settings: str | Path | None = None,
        add_dirs: Sequence[Path] = (),
        env: dict[str, str] | None = None,
        transcript_path: Path | None = None,
        **_: Any,  # `phase`, which only FakeHarness needs
    ) -> HarnessResult:
        argv: list[str] = [self.binary, "-p", "--output-format", "json"]
        argv += ["--model", model, "--effort", effort]

        if tools is not None:
            argv += ["--tools", ",".join(tools) if tools else ""]
        if schema is not None:
            argv += ["--json-schema", json.dumps(schema)]
        if resume:
            argv += ["--resume", resume]
            if fork:
                argv.append("--fork-session")
        elif session_id:
            argv += ["--session-id", session_id]
        if permission_mode:
            argv += ["--permission-mode", permission_mode]
        if max_budget_usd:
            argv += ["--max-budget-usd", str(max_budget_usd)]
        if append_system_prompt:
            argv += ["--append-system-prompt", append_system_prompt]
        if settings:
            argv += ["--settings", str(settings)]
        if not self.inherit_user_settings:
            argv += ["--setting-sources", ""]
        for d in add_dirs:
            argv += ["--add-dir", str(d)]

        child_env = {**os.environ, **(env or {})}
        # A headless child launched from inside a Claude Code session inherits
        # markers that make it behave as a nested session.  Cells must run
        # identically whether the operator starts them from a terminal or from
        # inside an agent, so the markers are stripped.
        for marker in ("CLAUDECODE", "CLAUDE_CODE_SSE_PORT", "CLAUDE_CODE_ENTRYPOINT"):
            child_env.pop(marker, None)
        started = time.time()
        try:
            proc = subprocess.run(
                argv,
                input=prompt,
                cwd=str(cwd),
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            raise HarnessError(
                f"agent call exceeded {timeout_s:.0f}s (model={model}, cwd={cwd})"
            ) from None
        duration = time.time() - started

        if transcript_path:
            Path(transcript_path).parent.mkdir(parents=True, exist_ok=True)
            Path(transcript_path).write_text(
                json.dumps(
                    {
                        "argv": argv[:2] + ["<prompt via stdin>"] + argv[2:],
                        "prompt": prompt,
                        "stdout": proc.stdout,
                        "stderr": proc.stderr[-4000:],
                        "returncode": proc.returncode,
                        "duration_s": duration,
                    },
                    indent=2,
                )
            )

        if not proc.stdout.strip():
            raise HarnessError(
                f"agent produced no output (rc={proc.returncode}): "
                f"{proc.stderr.strip()[-600:]}"
            )

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise HarnessError(
                f"could not parse agent output: {exc}\n{proc.stdout[:600]}"
            ) from exc

        structured = payload.get("structured_output")
        if structured is None and schema is not None:
            # The CLI validated against the schema, so the text is the object.
            try:
                structured = json.loads(payload.get("result", ""))
            except (json.JSONDecodeError, TypeError):
                structured = None

        return HarnessResult(
            text=payload.get("result", "") or "",
            structured=structured,
            session_id=payload.get("session_id", ""),
            cost_usd=float(payload.get("total_cost_usd") or 0.0),
            num_turns=int(payload.get("num_turns") or 0),
            is_error=bool(payload.get("is_error")),
            duration_s=duration,
            usage=payload.get("usage") or {},
            model_usage=payload.get("modelUsage") or {},
            raw=payload,
        )


class OpenCodeHarness:
    """The mixed-model cell's seam.

    OpenCode headless is ``opencode run -m <provider/model> <prompt>``.  It has
    no equivalent of ``--json-schema``, so structured phases fall back to
    parsing the last JSON object out of the reply, and tool permissions come
    from an ``opencode.json`` dropped in the worktree rather than from flags.
    """

    name = "opencode"

    def __init__(self, binary: str = "opencode"):
        self.binary = binary

    def query(
        self,
        prompt: str,
        *,
        cwd: Path,
        model: str,
        timeout_s: float = 900,
        transcript_path: Path | None = None,
        env: dict[str, str] | None = None,
        **_: Any,
    ) -> HarnessResult:
        argv = [self.binary, "run", "-m", model, prompt]
        started = time.time()
        try:
            proc = subprocess.run(
                argv,
                cwd=str(cwd),
                env={**os.environ, **(env or {})},
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            raise HarnessError(f"opencode call exceeded {timeout_s:.0f}s") from None
        except FileNotFoundError:
            raise HarnessError(
                "opencode is not installed on this box; the mixed cell needs it"
            ) from None
        duration = time.time() - started

        if transcript_path:
            Path(transcript_path).parent.mkdir(parents=True, exist_ok=True)
            Path(transcript_path).write_text(
                json.dumps(
                    {"argv": argv[:4] + ["<prompt>"], "prompt": prompt,
                     "stdout": proc.stdout, "stderr": proc.stderr[-4000:],
                     "returncode": proc.returncode, "duration_s": duration},
                    indent=2,
                )
            )

        return HarnessResult(
            text=proc.stdout,
            structured=parse_trailing_json(proc.stdout),
            session_id="",
            # OpenCode does not report spend; the cell's dollar ceiling cannot
            # see these calls, so budget mixed cells by hand.
            cost_usd=0.0,
            num_turns=0,
            is_error=proc.returncode != 0,
            duration_s=duration,
        )


# Canned ideas for --fake-agents.  They name real knobs in the stand-in
# train.py so the edit the fake agent makes is a genuine diff.
_FAKE_IDEAS = [
    ("Raise the learning rate", "learning_rate", 0.03),
    ("Lower weight decay", "weight_decay", 0.05),
    ("Deepen the model", "depth", 10),
    ("Widen the residual stream", "n_embd", 640),
    ("Shorten warmup", "warmup_steps", 50),
    ("Smaller batches, more steps", "batch_size", 16),
    ("Raise Muon momentum", "muon_momentum", 0.98),
    ("A little dropout", "dropout", 0.05),
]


class FakeHarness:
    """Canned agent output, no API calls, for testing the loop itself.

    It is not a mock of the *harness* but of the *researcher*: the select phase
    really edits ``train.py``, really invokes ``arena-train``, and really writes
    ``candidate.json``, so worktrees, the GPU mutex, the run-count budget,
    selection and archiving are all exercised end to end for nothing.
    """

    name = "fake"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def query(
        self,
        prompt: str,
        *,
        cwd: Path,
        model: str = "fake",
        phase: str = "",
        env: dict[str, str] | None = None,
        transcript_path: Path | None = None,
        **_: Any,
    ) -> HarnessResult:
        started = time.time()
        with self._lock:
            title, knob, value = self._rng.choice(_FAKE_IDEAS)

        if phase == "propose":
            structured = {
                "title": title,
                "justification": f"canned fake-agent idea; sets {knob} to {value}",
                # Long enough to look like a real proposal: the orchestrator
                # flags contentless ones, and a fake agent that trips that
                # check would make every smoke run look degenerate.
                "detail": (
                    f"Edit train.py and set `{knob} = {value}` in the hyperparameter "
                    f"block near the top of the file, leaving every other constant "
                    f"unchanged so the effect of {knob} is isolated. This is canned "
                    f"output from the fake harness, not a real research idea."
                ),
                "notes": "",
            }
        elif phase == "respond":
            structured = {
                "response": f"Noted. I would watch {knob} for a VRAM regression.",
                "notes": "",
            }
        else:
            structured = None
            self._do_select(Path(cwd), title, knob, value, env or {})

        return HarnessResult(
            text=json.dumps(structured) if structured else "done",
            structured=structured,
            session_id=new_session_id(),
            cost_usd=0.0,
            num_turns=1,
            is_error=False,
            duration_s=time.time() - started,
        )

    def _do_select(self, worktree: Path, title: str, knob: str, value: Any,
                   env: dict[str, str]) -> None:
        train_py = worktree / "train.py"
        if train_py.exists():
            src = train_py.read_text()
            patched, n = re.subn(
                rf"^{re.escape(knob)} = .*$", f"{knob} = {value}", src, count=1, flags=re.M
            )
            if n:
                train_py.write_text(patched)

        subprocess.run(
            ["arena-train"],
            cwd=str(worktree),
            env={**dict(os.environ), **env},
            capture_output=True,
            text=True,
            timeout=900,
        )

        (worktree / "candidate.json").write_text(
            json.dumps(
                {
                    "title": title,
                    "description": f"{knob} = {value}",
                    "status": "ok",
                    "reflection": "fake agent; no reasoning to report",
                },
                indent=2,
            )
        )


def parse_trailing_json(text: str) -> dict | None:
    """Last JSON object in a reply, for harnesses without schema support."""
    if not text:
        return None
    for match in reversed(list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.S))):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def new_session_id() -> str:
    return str(uuid.uuid4())


def build_harness(name: str, seed: int = 0) -> Harness:
    if name == "claude_code":
        return ClaudeCodeHarness()
    if name == "opencode":
        return OpenCodeHarness()
    if name == "fake":
        return FakeHarness(seed=seed)
    raise ValueError(f"unknown harness {name!r}")
