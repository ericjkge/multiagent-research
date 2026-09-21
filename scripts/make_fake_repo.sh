#!/usr/bin/env bash
# A stand-in for karpathy/autoresearch so the orchestration can be tested on a
# laptop.  The training script is never executed in fake-GPU mode -- arena-train
# simulates it -- but it must be real enough that agents have something
# meaningful to read and edit.
set -euo pipefail

REPO="${1:-$HOME/.cache/arena-fake-autoresearch}"
rm -rf "$REPO"
mkdir -p "$REPO"
cd "$REPO"

cat > prepare.py <<'PY'
"""Fixed constants, data prep and the ground-truth evaluation. Read-only."""
SEQ_LEN = 1024
TRAIN_SECONDS = 300
VOCAB_SIZE = 50257


def get_dataloader(batch_size, seq_len=SEQ_LEN):
    raise NotImplementedError("provided by the real autoresearch repo")


def evaluate_bpb(model):
    """The metric. Vocabulary-size independent; lower is better."""
    raise NotImplementedError("provided by the real autoresearch repo")
PY

cat > train.py <<'PY'
"""GPT training loop. This is the only file the agent may modify."""
import math

from prepare import SEQ_LEN, TRAIN_SECONDS, VOCAB_SIZE, evaluate_bpb, get_dataloader

# --- hyperparameters -------------------------------------------------
depth = 8
n_head = 8
n_embd = 512
batch_size = 32
learning_rate = 0.02
weight_decay = 0.1
warmup_steps = 100
muon_momentum = 0.95
dropout = 0.0


def lr_schedule(step, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_model():
    """GPT with rotary embeddings, RMSNorm and a Muon+AdamW split."""
    raise NotImplementedError


def train():
    model = build_model()
    loader = get_dataloader(batch_size)
    # ... trains for exactly TRAIN_SECONDS, then evaluates
    return evaluate_bpb(model)


if __name__ == "__main__":
    train()
PY

cat > pyproject.toml <<'TOML'
[project]
name = "autoresearch"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["torch"]
TOML

cat > README.md <<'MD'
# autoresearch (local stand-in)

Fake substrate for smoke-testing the arena orchestrator without a GPU.
MD

git init -q
git add -A
git -c user.name=arena -c user.email=arena@local commit -qm "fake autoresearch baseline"
echo "fake repo ready at $REPO ($(git rev-parse --short HEAD))"
