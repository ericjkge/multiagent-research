#!/usr/bin/env bash
# End-to-end check of the orchestration with a simulated GPU.
#
#   bash scripts/smoke_test.sh --fake-agents                          # rounds protocol, free
#   bash scripts/smoke_test.sh                                        # rounds protocol, real Haiku agents
#   bash scripts/smoke_test.sh --config configs/smoke_open.yaml --fake-agents   # open protocol, free
#   bash scripts/smoke_test.sh --config configs/smoke_open.yaml       # open protocol, real Haiku agents
#
# All exercise worktrees, the GPU mutex, the run budget, selection or the
# score log, archiving and the analysis pipeline.  Only the paid ones exercise
# the prompts, structured output, session forking/resuming and the guard hook.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/smoke.yaml"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

bash scripts/make_fake_repo.sh
RUN_DIR="runs/$(basename "$CONFIG" .yaml)-$(date +%Y%m%d-%H%M%S)"
python3 -m orchestrator.run --config "$CONFIG" --run-dir "$RUN_DIR" "${ARGS[@]+"${ARGS[@]}"}"
python3 -m analysis.report "$RUN_DIR"
python3 -m analysis.verify "$RUN_DIR"
