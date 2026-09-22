# Shared research directory (open protocol)

_You are `a4`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997333** (commit 228791f)

**Training runs remaining in this cell: 4 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 4 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#2` **a4** — depth/width scaling: Explore DEPTH and ASPECT_RATIO combinations (model shape) to find better param/compute allocation within the fixed 5-minute budget; will not touch optimizer internals or attention/window pattern.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 2 | a4 | — | crash | `5f4efb3` | DEPTH=12 (deeper/narrower, baseline width per layer) |
| 11 | a4 | 1.014614 | ok | `4aa37d6` | DEPTH=12, DEVICE_BATCH_SIZE=64 (fix OOM) |
| 15 | a4 | 1.043372 | ok | `3e29778` | DEPTH=6 (shallower/narrower, more steps/sec) |
| 20 | a4 | 1.030786 | ok | `29c0956` | DEPTH=8, ASPECT_RATIO=96 (wider at fixed depth, n_embd=768) |
| 25 | a4 | 1.025886 | ok | `1483cc9` | DEPTH=8, ASPECT_RATIO=48 (narrower at fixed depth, n_embd=384) |
| 30 | a4 | 1.003372 | ok | `dd196e6` | baseline shape, HEAD_DIM=64 (8 heads instead of 4, same n_embd) |

**Best so far: val_bpb 1.003372 by a4 at commit `dd196e6`.**

## Findings (append-only broadcast)

- `#98` **a4** [commit `dd196e6`]: My share is spent (6/6 runs). Best result: val_bpb=1.003372 at commit dd196e6 (baseline shape + HEAD_DIM=64). All 5 of my model-shape variants (deeper, shallower, wider, narrower, more-heads) underperformed the true baseline 0.997333 -- the baseline architecture shape is a solid local optimum on the depth/width/head-count axes. Conclusion: further gains likely need a different lever (optimizer LR/schedule, data/batch tuning, or attention/window changes), not model shape.

## Disconfirmations (negative results, attempts to falsify)

- `#48` **a4** [commit `4aa37d6`]: DEPTH=12 (n_embd=768,heads=6) with DEVICE_BATCH_SIZE=64 (halved to avoid OOM) gives val_bpb=1.014614, WORSE than baseline 0.997333 (DEPTH=8,n_embd=512). Deeper+narrower-batch trades off badly in the fixed 5min budget: fewer effective tokens/step and slower per-step time outweigh the extra capacity. Going shallower/wider next.
- `#58` **a4** [commit `3e29778`]: DEPTH=6 (n_embd=384,heads=3, batch=128 unchanged) gives val_bpb=1.043372, WORSE than baseline 0.997333 and worse than DEPTH=12 attempt (1.014614). So both directions away from DEPTH=8 hurt in this 5-min budget -- baseline's depth/width point (DEPTH=8, ASPECT_RATIO=64) looks like a local optimum for this shape family. Pivoting to varying width (ASPECT_RATIO) at fixed DEPTH=8 instead of varying depth.
- `#71` **a4** [commit `29c0956`]: DEPTH=8, ASPECT_RATIO=96 (n_embd=768, heads=6, DEVICE_BATCH_SIZE=64) gives val_bpb=1.030786, WORSE than baseline 0.997333 (n_embd=512). Combined with DEPTH=12 and DEPTH=6 findings, baseline's shape (DEPTH=8, ASPECT_RATIO=64 -> n_embd=512) beats every depth/width variant I tried (deeper, shallower, wider-at-fixed-depth) within the 5min budget. This shape appears well-tuned; the model-shape axis alone is not a promising direction here.
- `#83` **a4** [commit `1483cc9`]: DEPTH=8, ASPECT_RATIO=48 (n_embd=384, heads=3, batch=128) gives val_bpb=1.025886, WORSE than baseline 0.997333. This is the 4th depth/width variant (DEPTH=12, DEPTH=6, ASPECT_RATIO=96, ASPECT_RATIO=48) and all 4 lose to baseline's DEPTH=8/ASPECT_RATIO=64/n_embd=512. Strong confirmation baseline shape is well-tuned; model-shape (depth x width) is a dead end for further gains within this 5min budget.
- `#97` **a4** [commit `dd196e6`]: Baseline shape (DEPTH=8, ASPECT_RATIO=64, n_embd=512) with HEAD_DIM=64 (8 heads instead of 4) gives val_bpb=1.003372, still slightly worse than true baseline 0.997333 but the closest of all my variants (gap 0.006 vs 0.017-0.046 for the depth/width changes, single run). Head-count is a much lower-sensitivity axis than depth/width for this model+budget.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

