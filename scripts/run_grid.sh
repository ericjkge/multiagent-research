#!/usr/bin/env bash
# Run several cell chains in parallel on one multi-GPU box, one chain per GPU.
#
#   bash scripts/run_grid.sh "open_opus_6 indep_opus_6" "opus_6 opus_1" "open_opus_1 indep_sonnet_6" "indep_haiku_6"
#   GPU_START=2 bash scripts/run_grid.sh "..." "..."     # start at GPU 2 (a second launch, e.g. different auth)
#
# Chain i runs on GPU i (CUDA_VISIBLE_DEVICES=i), sequentially, via run_cells.sh, which verifies
# and archives every cell to results/.  Cells on different GPUs never share a card, so the
# single-GPU invariant holds per cell.  Logs: runs/gpu<i>.out.  Needs setup_gpu_box.sh done once.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
n_gpu=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
i=${GPU_START:-0}
[ $((i + $#)) -le "$n_gpu" ] || { echo "$# chains from GPU $i but only $n_gpu GPUs" >&2; exit 1; }
for chain in "$@"; do
  echo "GPU $i: $chain"
  CUDA_VISIBLE_DEVICES=$i nohup bash scripts/run_cells.sh $chain > "runs/gpu$i.out" 2>&1 &
  i=$((i + 1))
  sleep 20   # stagger so the data cache and the first API calls do not all hit at once
done
echo "launched $i chain(s); watch with: tail -f runs/gpu*.out"
