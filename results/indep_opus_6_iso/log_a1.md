# Shared research directory (open protocol)

_You are `a1`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#3` **a1** — optimization scale (batch size x step count x LR schedule): I will treat the token budget as fixed and re-allocate it: shrink TOTAL_BATCH_SIZE to buy more optimizer steps, then retune Muon/Adam LR and the warmdown shape to match. I will NOT assume the baseline 2**19-token batch or DEPTH=8 are near-optimal, and I will not chase architecture novelties (SwiGLU/MoE/new attention variants).

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 1 | a1 | 1.001379 | ok | `414957f` | halve TOTAL_BATCH_SIZE to 2**18 (2x optimizer steps, grad_accum=1) |
| 5 | a1 | 1.008119 | ok | `33e4268` | quarter batch: TOTAL_BATCH_SIZE 2**17, device batch 64 |
| 9 | a1 | 0.997025 | ok | `1916b20` | flat-concat fast dataloader in train.py (removes CPU packing bottleneck), batch 2**18 |
| 13 | a1 | 0.994308 | ok | `0e6a355` | DEPTH 8 -> 10 at fixed model_dim 512 (more capacity now that throughput is higher) |
| 17 | a1 | 0.995232 | ok | `f8923b5` | DEPTH 10 -> 12 at fixed model_dim 512 |
| 21 | a1 | 0.996529 | ok | `a57aebd` | back to DEPTH 10; WARMDOWN_RATIO 0.5 -> 0.35 (more time at full LR) |

**Best so far: val_bpb 0.994308 by a1 at commit `0e6a355`.**

## Findings (append-only broadcast)

- `#7` **a1** [commit `414957f`]: Halving TOTAL_BATCH_SIZE 2**19 -> 2**18 (grad_accum 2 -> 1, ~2x optimizer steps in the same 300s) gives val_bpb 1.001379 vs baseline 1.012347 (-0.011). LRs unchanged. The baseline batch is above the critical batch size for this ~50M model.
- `#23` **a1** [commit `1916b20`]: Replacing prepare.make_dataloader with a flat document-concatenation loader written in train.py (array('q') pool -> torch.frombuffer -> pinned -> GPU, ~109ms CPU/batch vs ~209ms) gives val_bpb 0.997025 at batch 2**18, vs 1.001379 with the same batch and the stock loader (-0.0044). The baseline run is dataloader-bound, not GPU-bound; neither loader masks attention across doc boundaries so packing style is otherwise equivalent.
- `#31` **a1** [commit `0e6a355`]: DEPTH 8 -> 10 at fixed model_dim 512 (ASPECT_RATIO 51) improves val_bpb 0.997025 -> 0.994308 with the fast loader and batch 2**18. Extra capacity pays off now that throughput is ~40% higher (474M tokens at depth 8).
- `#48` **a1** [commit `0e6a355`]: a1 final: best val_bpb 0.994308 (commit 0e6a355) = baseline 1.012347 minus 0.018. Recipe: TOTAL_BATCH_SIZE 2**18 (device batch 128, grad_accum 1) + flat-concat dataloader written in train.py instead of prepare.make_dataloader (the stock Python best-fit packer is the throughput bottleneck: ~209ms CPU vs ~180ms GPU per 128x2048 batch; MFU 28% -> 40%) + DEPTH 10 at model_dim 512 (ASPECT_RATIO 51). Everything else is baseline. Batch 2**17, DEPTH 12 and WARMDOWN_RATIO 0.35 were each measured worse.

## Disconfirmations (negative results, attempts to falsify)

- `#22` **a1** [commit `33e4268`]: Quartering the batch (2**17, device batch 64) is WORSE than 2**18: val_bpb 1.008119 vs 1.001379. 2**18 is the sweet spot; part of the loss at 2**17 is throughput, not optimization (the Python best-fit dataloader costs ~209ms CPU per 128x2048 batch and caps tok/s).
- `#39` **a1** [commit `f8923b5`]: DEPTH 12 at model_dim 512 (ASPECT_RATIO 42) does NOT continue the depth trend: val_bpb 0.995232 vs 0.994308 at DEPTH 10. Depth ~10 at width 512 is the optimum of this axis for a 300s budget with the fast loader and batch 2**18.
- `#47` **a1** [commit `a57aebd`]: Shortening the LR warmdown at DEPTH 10 (WARMDOWN_RATIO 0.5 -> 0.35, more time at full LR) is worse: val_bpb 0.996529 vs 0.994308 with 0.5. The long linear-to-zero decay over the last half of the budget is the better shape; do not trade annealing time for high-LR time here.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

