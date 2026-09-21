#!/usr/bin/env bash
# End-to-end check of the orchestration with a simulated GPU.
#
#   bash scripts/smoke_test.sh --fake-agents   # free: no API calls at all
#   bash scripts/smoke_test.sh                 # a few cents: real Haiku agents
#
# Both exercise worktrees, the GPU mutex, the run budget, best-of-N, selection,
# archiving and the analysis pipeline.  Only the paid one exercises the prompts,
# structured output, session forking and the guard hook.
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/make_fake_repo.sh
RUN_DIR="runs/smoke-$(date +%Y%m%d-%H%M%S)"
python3 -m orchestrator.run --config configs/smoke.yaml --run-dir "$RUN_DIR" "$@"
python3 -m analysis.report "$RUN_DIR"
python3 -m analysis.verify "$RUN_DIR"
