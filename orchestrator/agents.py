"""Thin wrappers around coding-agent CLIs in headless mode.

claude:   `claude -p` (Claude Code). Model ids: claude-haiku-4-5-20251001, claude-sonnet-5, claude-opus-5.
opencode: `opencode run` (stub, same contract). Fill in once someone has OpenCode installed on the GPU box.
fake:     no API call, returns canned JSON. For testing the loop.

Contract: run_agent(kind, model, cwd, prompt, allow_edits, timeout) -> (text, meta)
The last line of `text` should be a JSON object; parse_json() extracts it.
"""
import json
import os
import re
import subprocess
import time

READ_TOOLS = "Read,Grep,Glob,LS"
EDIT_TOOLS = "Read,Grep,Glob,LS,Edit,Write,MultiEdit"


def run_agent(kind, model, cwd, prompt, allow_edits=False, timeout=900, effort=None):
    if kind == "fake":
        return _fake(prompt, cwd, allow_edits), {"kind": "fake"}
    if kind == "claude":
        return _claude(model, cwd, prompt, allow_edits, timeout, effort)
    if kind == "opencode":
        return _opencode(model, cwd, prompt, allow_edits, timeout)
    raise ValueError(kind)


def _claude(model, cwd, prompt, allow_edits, timeout, effort):
    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "json",
        "--max-turns", "25" if allow_edits else "8",
        "--allowedTools", EDIT_TOOLS if allow_edits else READ_TOOLS,
        "--permission-mode", "acceptEdits" if allow_edits else "default",
    ]
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)  # allow nesting when launched from inside Claude Code
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    meta = {"kind": "claude", "model": model, "seconds": round(time.time() - t0, 1), "rc": p.returncode}
    text = p.stdout
    try:
        j = json.loads(p.stdout)
        text = j.get("result", "")
        if j.get("is_error"):
            meta["error"] = text[:300]  # e.g. 401 OAuth token expired: run `claude` and log in, or set ANTHROPIC_API_KEY
        meta["cost_usd"] = j.get("total_cost_usd")
        meta["turns"] = j.get("num_turns")
        meta["usage"] = j.get("usage")
    except Exception:
        meta["stderr"] = p.stderr[-2000:]
    return text, meta


def _opencode(model, cwd, prompt, allow_edits, timeout):
    # OpenCode headless: `opencode run -m <provider/model> "<prompt>"`. Tool permissions are set in
    # opencode.json in the worktree; for propose/respond steps use a read-only agent config.
    cmd = ["opencode", "run", "-m", model, prompt]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.stdout, {"kind": "opencode", "model": model, "seconds": round(time.time() - t0, 1), "rc": p.returncode}


_FAKE_IDEAS = [
    "Raise MATRIX_LR from 0.04 to 0.06", "Lower WEIGHT_DECAY to 0.1", "Set DEPTH to 10",
    "Use WARMDOWN_RATIO 0.6", "Reduce DEVICE_BATCH_SIZE to 64 to fit more steps",
    "Change ADAM_BETAS to (0.9, 0.95)", "Increase HEAD_DIM to 256", "Set EMBEDDING_LR to 0.4",
]


def _fake(prompt, cwd, allow_edits):
    import random
    if '"ideas"' in prompt:
        k = int(re.search(r"Propose (\d+) DIFFERENT", prompt).group(1))
        ideas = random.sample(_FAKE_IDEAS, k)
        return json.dumps({"ideas": [{"idea": i, "justification": "fake"} for i in ideas]})
    if "Step 1 (PROPOSE)" in prompt:
        return json.dumps({"idea": random.choice(_FAKE_IDEAS), "justification": "fake"})
    if "Step 2 (RESPOND)" in prompt:
        return json.dumps({"response": "Fine by me, but watch for divergence."})
    # finalize / implement: make a tiny real edit so the diff machinery is exercised
    if allow_edits:
        path = os.path.join(cwd, "train.py")
        src = open(path).read()
        m = re.search(r"^MATRIX_LR = ([0-9.]+)", src, re.M)
        if m:
            new = round(float(m.group(1)) * random.choice([0.8, 1.25]), 4)
            src = src.replace(m.group(0), f"MATRIX_LR = {new}", 1)
            open(path, "w").write(src)
    return json.dumps({"final_idea": "fake change to MATRIX_LR", "changed_from_proposal": False,
                       "predicted_val_bpb": 0.99, "done": True})


def parse_json(text):
    """Return the last JSON object in text, or None."""
    if not text:
        return None
    for m in reversed(list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.S))):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None
