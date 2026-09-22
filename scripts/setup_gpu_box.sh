#!/usr/bin/env bash
# One-time setup on a rented H100 box.
#
# Bootstraps the toolchain, clones autoresearch, prepares the data, and
# measures the baseline ONCE.  Every cell run on this box reuses that baseline,
# so all cells start from an identical number and none of them spends five
# minutes re-measuring it.
#
# Tested target: 1x H100 80GB, Ubuntu, CUDA image (RunPod "pytorch" template).
#
#   bash scripts/setup_gpu_box.sh              # full setup
#   bash scripts/setup_gpu_box.sh --noise-gate # ALSO run the noise gate (+30 min)
set -euo pipefail

REPO="${AUTORESEARCH_DIR:-$HOME/autoresearch}"
UPSTREAM="https://github.com/karpathy/autoresearch"
PIN="${AUTORESEARCH_COMMIT:-}"
NOISE_GATE=0
[ "${1:-}" = "--noise-gate" ] && NOISE_GATE=1

# -- toolchain ---------------------------------------------------------------
# Fresh cloud boxes have none of this.  Skipped silently when already present,
# so the script is safe to re-run.
if ! command -v git >/dev/null || ! command -v curl >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq git curl nodejs npm >/dev/null
fi

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

command -v claude >/dev/null || npm install -g @anthropic-ai/claude-code
claude --version

# The orchestrator's own dependencies (analysis needs scikit-learn and matplotlib).
python3 -c "import yaml, sklearn, matplotlib" 2>/dev/null || \
  pip install -q pyyaml scikit-learn matplotlib 2>/dev/null || \
  pip install -q --break-system-packages pyyaml scikit-learn matplotlib

# Agent calls need an API key, not a claude.ai login: the cells run headless
# and unattended, and an expired OAuth token fails every call in the cell.
[ -n "${ANTHROPIC_API_KEY:-}" ] || echo "WARNING: ANTHROPIC_API_KEY is unset; agent calls will fail"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# -- substrate ---------------------------------------------------------------
if [ ! -d "$REPO/.git" ]; then
  git clone "$UPSTREAM" "$REPO"
fi
cd "$REPO"
git fetch --all --quiet
[ -n "$PIN" ] && git checkout -q "$PIN"
COMMIT=$(git rev-parse HEAD)

uv sync
[ -d "$HOME/.cache/autoresearch" ] || uv run prepare.py

# -- the baseline, measured once ---------------------------------------------
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

# -- the noise gate ----------------------------------------------------------
# Run-to-run spread on identical code.  Any "improvement" smaller than this is
# indistinguishable from noise, and since every cell is scored by single runs,
# a wide spread invalidates the whole grid rather than one cell.  Measure it
# once, on this box, and put the number on the slide.
if [ "$NOISE_GATE" = "1" ]; then
  if [ ! -f noise_gate.json ]; then
    echo "noise gate: 5 more baseline runs (~25 minutes) ..."
    VALS="$VAL"
    for i in 2 3 4 5 6; do
      uv run train.py > "noise_$i.log" 2>&1 || true
      V=$(grep '^val_bpb:' "noise_$i.log" | awk '{print $2}')
      [ -n "$V" ] && VALS="$VALS $V"
      echo "  run $i: ${V:-FAILED}"
    done
    python3 - "$COMMIT" $VALS > noise_gate.json <<'PY'
import json, statistics, sys
commit, vals = sys.argv[1], [float(v) for v in sys.argv[2:]]
spread = max(vals) - min(vals)
print(json.dumps({
    "commit": commit, "n": len(vals), "vals": vals,
    "mean": round(statistics.mean(vals), 6),
    "stdev": round(statistics.stdev(vals), 6) if len(vals) > 1 else None,
    "spread": round(spread, 6),
    "verdict": "OK" if spread <= 0.003 else "TOO NOISY",
}, indent=2))
PY
  fi
  cat noise_gate.json
  python3 -c "
import json,sys
g=json.load(open('noise_gate.json'))
print()
if g['verdict']=='OK':
    print(f\"noise gate PASSED: spread {g['spread']:.6f} <= 0.003.\")
    print('Improvements smaller than that are still not results. Quote it on the slide.')
else:
    print(f\"noise gate FAILED: spread {g['spread']:.6f} > 0.003.\")
    print('Single runs cannot distinguish ideas on this box. Either lengthen runs')
    print('(TIME_BUDGET in prepare.py, the SAME value on every box and every cell)')
    print('or accept it and say so explicitly when presenting.')
"
else
  echo
  echo "Noise gate NOT measured. Run 'bash scripts/setup_gpu_box.sh --noise-gate'"
  echo "before the first real cell -- without it you cannot tell a result from noise."
fi

echo
echo "Pin this in every config:  autoresearch_commit: \"$COMMIT\""
