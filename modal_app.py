"""Run a cell on Modal: one Function per cell, one dedicated H100 each.

Everything that needs flock (gpu.lock, log.jsonl, budget.json) stays on the
container's local disk -- Modal Volumes don't support file locking and are
last-write-wins on concurrent writes to the same file, which would silently
corrupt those. The Volume only holds things nothing writes concurrently: the
prepared autoresearch clone (read once, copied locally) and archived results
(written once, at the end).

Usage:
    modal run modal_app.py --cells open_sonnet_1
    modal run modal_app.py --cells open_sonnet_1,open_sonnet_3,open_sonnet_6

Needs a Modal secret named "anthropic-api-key" with ANTHROPIC_API_KEY set.
"""

import os
import subprocess
import time
from pathlib import Path

import modal

app = modal.App("multiagent-research")
volume = modal.Volume.from_name("multiagent-research", create_if_missing=True)
VOL = "/vol"

AUTORESEARCH_URL = "https://github.com/karpathy/autoresearch"

# Modal containers run as root, and claude refuses --dangerously-skip-permissions
# under root. Everything except the golden-repo prep runs as this user instead.
RUNNER = "runner"
HOME = f"/home/{RUNNER}"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "nodejs", "npm", "rsync")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "npm install -g @anthropic-ai/claude-code",
        f"useradd -m -d {HOME} {RUNNER}",
    )
    .pip_install("pyyaml")
    .add_local_dir(".", remote_path=f"{HOME}/arena", ignore=["runs", ".git", "__pycache__"])
)


def sh(cmd: str, cwd: str | None = None) -> str:
    proc = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed: {cmd}\n{proc.stderr}")
    return proc.stdout.strip()


def prepare_golden_repo() -> str:
    """Clone + prepare + measure baseline once, persisted on the Volume.

    prepare.py writes its data/tokenizer cache to ``~/.cache/autoresearch``,
    which is HOME-relative, not repo-relative -- so it never lands on the
    Volume by itself, and each container starts with a cold cache. Persist it
    separately at /vol/autoresearch-cache so run_cell can restore it for
    whichever user actually trains.
    """
    golden = f"{VOL}/autoresearch-golden"
    cache = f"{VOL}/autoresearch-cache"

    if not (Path(golden) / "baseline.json").exists():
        sh(f"git clone {AUTORESEARCH_URL} {golden}")
        sh("uv sync", cwd=golden)
        sh("uv run prepare.py", cwd=golden)
        sh(f"cp -r ~/.cache/autoresearch {cache}")

        out = sh("uv run train.py", cwd=golden)
        val = next(l.split()[1] for l in out.splitlines() if l.startswith("val_bpb:"))
        vram = next(
            (l.split()[1] for l in out.splitlines() if l.startswith("peak_vram_mb:")), "0"
        )
        commit = sh("git rev-parse HEAD", cwd=golden)
        (Path(golden) / "baseline.json").write_text(
            f'{{"commit": "{commit}", "val_bpb": {val}, "peak_vram_mb": {vram}}}\n'
        )
        volume.commit()
    elif not Path(cache).exists():
        # baseline.json predates the cache fix (or the cache was lost) --
        # rebuild just the cache, without re-measuring the baseline.
        sh("uv run prepare.py", cwd=golden)
        sh(f"cp -r ~/.cache/autoresearch {cache}")
        volume.commit()

    return golden


@app.function(
    image=image,
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={VOL: volume},
    secrets=[modal.Secret.from_name("anthropic-api-key")],
)
def run_cell(cell: str) -> dict:
    golden = prepare_golden_repo()

    local_repo = f"{HOME}/autoresearch"
    sh(f"cp -r {golden} {local_repo}")
    sh(f"mkdir -p {HOME}/.cache && cp -r {VOL}/autoresearch-cache {HOME}/.cache/autoresearch")
    sh(f"chown -R {RUNNER}:{RUNNER} {HOME}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = f"{HOME}/runs/{cell}-{stamp}"
    dest = f"{VOL}/runs/{cell}-{stamp}"

    def checkpoint() -> None:
        if not Path(run_dir).exists():
            return
        # Exclude work/: live git worktrees with their own .venv (numpy,
        # torch, CUDA libs -- gigabytes per agent). That's what filled the
        # Volume and crashed a checkpoint before. Everything else in run_dir
        # (log.jsonl, transcripts/, rounds/, state.json, ...) is small and
        # is the actual record worth keeping.
        sh(f"mkdir -p {VOL}/runs && rm -rf {dest} && "
           f"rsync -a --exclude=work {run_dir}/ {dest}/")
        volume.commit()

    # Non-blocking + inherited stdout/stderr, so `modal app logs` streams live
    # instead of being blind until the whole (possibly hours-long) run ends.
    proc = subprocess.Popen(
        [
            "python3", "-m", "orchestrator.run",
            "--config", f"configs/{cell}.yaml",
            "--repo", local_repo,
            "--run-dir", run_dir,
        ],
        cwd=f"{HOME}/arena",
        user=RUNNER,
        env={**os.environ, "HOME": HOME, "USER": RUNNER},
    )

    # Checkpoint every 5 min so a container interruption never loses more than
    # that much progress -- the run itself is unaffected either way.
    try:
        while proc.poll() is None:
            time.sleep(300)
            checkpoint()
    finally:
        checkpoint()

    return {"cell": cell, "run_dir": dest, "returncode": proc.returncode}


@app.local_entrypoint()
def main(cells: str):
    names = [c.strip() for c in cells.split(",") if c.strip()]
    calls = [run_cell.spawn(c) for c in names]
    for c, call in zip(names, calls):
        print(f"{c}: {call.get()}")
