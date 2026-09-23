# Shared research directory (open protocol)

_You are `a0`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#2` **a0** — batch-size / update-count economics: Explore the number of optimizer updates per 5 minutes: shrink tokens-per-step (TOTAL_BATCH_SIZE, grad-accum) and co-tune LR/schedule so the same wall clock buys many more updates. I will not assume the baseline 524K-token batch or its LRs are near-optimal, and I will keep the architecture fixed while testing this axis.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 2 | a0 | 1.001892 | ok | `8a7f7ef` | halve tokens-per-step: TOTAL_BATCH_SIZE 2**19 -> 2**18 (grad_accum 1), LRs unchanged |
| 6 | a0 | 1.007828 | ok | `65ff337` | quarter tokens-per-step: TOTAL_BATCH_SIZE 2**17, device batch 64 (grad_accum 1), LRs unchanged |
| 10 | a0 | 1.002599 | ok | `4b786d2` | at 2**18 batch: scale all LRs ~1.4x (matrix 0.055, embed 0.85, unembed 0.0055) |
| 14 | a0 | 0.986791 | ok | `62ceae8` | bigger model at the better batch: DEPTH 8->10 (dim 512->640), TOTAL_BATCH_SIZE 2**18, baseline LRs |
| 18 | a0 | 0.994629 | ok | `7043cd0` | push model size: DEPTH 12 (dim 768), TOTAL_BATCH_SIZE 2**18 via device batch 64 x grad_accum 2 |
| 22 | a0 | 0.987351 | ok | `14f293f` | DEPTH 10 with TOTAL_BATCH_SIZE 2**17 (device batch 64, grad_accum 1): restore ~2000 updates at the larger model |

**Best so far: val_bpb 0.986791 by a0 at commit `62ceae8`.**

## Findings (append-only broadcast)

- `#9` **a0** [commit `8a7f7ef`]: Halving tokens-per-step (TOTAL_BATCH_SIZE 2**19->2**18, grad_accum 2->1, LRs unchanged) gives val_bpb 1.001892 vs baseline 1.012347 (-0.0105). The baseline batch is too large: 2x more optimizer updates in the same 5 min is worth ~0.01 bpb.
- `#25` **a0** _(weak claim)_ [commit `4b786d2`]: LRs are not the explanation for the batch-size win: scaling all LRs ~1.4x (matrix 0.055 / embed 0.85 / unembed 0.0055) at TOTAL_BATCH_SIZE 2**18 gives 1.002599 vs 1.001892 at the baseline LRs (matrix 0.04 / embed 0.6 / unembed 0.004). Flat-to-slightly-worse, so the baseline LRs are already near-optimal at 2**18.
- `#33` **a0** [commit `62ceae8`]: The baseline model is too small for the 5-minute budget: DEPTH 8->10 (dim 512->640, 5 heads x 128) at TOTAL_BATCH_SIZE 2**18 gives val_bpb 0.986791, -0.0151 vs 1.001892 (depth 8, same batch) and -0.0256 vs the 1.012347 baseline. Peak VRAM 67.8 GB at device batch 128.
- `#51` **a0** _(weak claim)_ [commit `14f293f`]: At DEPTH 10, TOTAL_BATCH_SIZE 2**17 (device batch 64, grad_accum 1) gives 0.987351 vs 0.986791 at 2**18 with device batch 128 - a tie, but peak VRAM halves (34.3 vs 67.8 GB). Useful if VRAM headroom is needed for a wider model.
- `#52` **a0** [commit `62ceae8`]: FINAL (a0): best val_bpb 0.986791 at commit 62ceae8 = DEPTH 10 (dim 640, 5 heads x 128) + TOTAL_BATCH_SIZE 2**18 (device batch 128, grad_accum 1), everything else as baseline. That is -0.0256 vs the 1.012347 baseline, from two independent one-line changes: half the tokens-per-step (-0.0105) and a deeper/wider model (-0.0151). Bounds found: tokens-per-step 2**17 is worse at depth 8, DEPTH 12 is worse at 2**18, and a 1.4x LR scale-up is flat.

## Disconfirmations (negative results, attempts to falsify)

- `#16` **a0** [commit `65ff337`]: Shrinking tokens-per-step further does NOT keep helping: 2**17 (device batch 64, LRs unchanged) gives val_bpb 1.007828, clearly worse than 2**18's 1.001892 (and barely better than the 2**19 baseline 1.012347). Optimum tokens-per-step is near 2**18 at these LRs; peak VRAM drops 44.9->22.7 GB.
- `#41` **a0** [commit `7043cd0`]: Model size does not keep paying: DEPTH 12 (dim 768, device batch 64 x grad_accum 2, TOTAL_BATCH_SIZE 2**18) gives 0.994629, worse than DEPTH 10's 0.986791. Optimum depth is ~10 at dim 640 for the 5-minute budget.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

