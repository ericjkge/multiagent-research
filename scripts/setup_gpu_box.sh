#!/usr/bin/env bash
# One-time setup on a rented H100 box.
#
# Clones autoresearch, prepares the data, and measures the baseline ONCE.  Every
# cell run on this box reuses that baseline, so all cells start from an
# identical number and none of them spends five minutes re-measuring it.
set -euo pipefail

REPO="${AUTORESEARCH_DIR:-$HOME/autoresearch}"
UPSTREAM="https://github.com/karpathy/autoresearch"
PIN="${AUTORESEARCH_COMMIT:-}"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v claude >/dev/null || { echo "install Claude Code first"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -d "$REPO/.git" ]; then
  git clone "$UPSTREAM" "$REPO"
fi
cd "$REPO"
git fetch --all --quiet
[ -n "$PIN" ] && git checkout -q "$PIN"
COMMIT=$(git rev-parse HEAD)

uv sync
[ -d "$HOME/.cache/autoresearch" ] || uv run prepare.py

if [ ! -f baseline.json ]; then
  echo "measuring the baseline (~5 minutes) ..."
  uv run train.py > baseline.log 2>&1
  VAL=$(grep '^val_bpb:' baseline.log | awk '{print $2}')
  VRAM=$(grep '^peak_vram_mb:' baseline.log | awk '{print $2}')
  [ -n "$VAL" ] || { echo "baseline run failed; see $REPO/baseline.log"; exit 1; }
  printf '{\n  "commit": "%s",\n  "val_bpb": %s,\n  "peak_vram_mb": %s\n}\n' \
    "$COMMIT" "$VAL" "$VRAM" > baseline.json
fi

cat baseline.json
echo
echo "Pin this in every config:  autoresearch_commit: \"$COMMIT\""
