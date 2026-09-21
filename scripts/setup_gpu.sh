#!/usr/bin/env bash
# One-time setup on a fresh GPU box (tested target: 1x H100 80GB, Ubuntu, CUDA image, e.g. RunPod "pytorch" template).
# Usage: bash scripts/setup_gpu.sh        (run from the multiagent-research checkout)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git curl nodejs npm > /dev/null

# uv (python + deps for autoresearch)
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH")
export PATH="$HOME/.local/bin:$PATH"

# Claude Code headless agent. Needs ANTHROPIC_API_KEY in the environment (an API key, not a Claude.ai login).
command -v claude >/dev/null || npm install -g @anthropic-ai/claude-code
claude --version

# autoresearch + data (downloads ClimbMix shards + trains the tokenizer into ~/.cache/autoresearch; ~10 min)
mkdir -p work
[ -d work/autoresearch ] || git clone https://github.com/karpathy/autoresearch.git work/autoresearch
cd work/autoresearch
uv sync
uv run prepare.py
cd ../..

# smoke: one real 5-minute run so we know the box works and how many steps it does
cd work/autoresearch && timeout 900 uv run train.py | tail -12; cd ../..
echo "SETUP OK. Now: export ANTHROPIC_API_KEY=...; python3 orchestrator/run.py --cell configs/cells.json:haiku_3 --out runs/haiku_3"
