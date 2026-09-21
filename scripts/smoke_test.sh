#!/usr/bin/env bash
# End-to-end check of the orchestration with a simulated GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/make_fake_repo.sh
RUN_DIR="runs/smoke-$(date +%Y%m%d-%H%M%S)"
python3 -m orchestrator.run --config configs/smoke.yaml --run-dir "$RUN_DIR"
python3 -m analysis.report "$RUN_DIR"
python3 -m analysis.verify "$RUN_DIR"
