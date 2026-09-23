# Shared research directory (open protocol)

_You are `a4`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997359** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#5` **a4** — model depth/width scaling under fixed 5-min budget: I will sweep DEPTH (and derived width/heads) to find the best capacity-vs-speed tradeoff at fixed wall-clock budget, plus try one optimizer/schedule tweak. I will not assume the baseline DEPTH=8 is already optimal for this time budget.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 4 | a4 | 1.002419 | ok | `f461662` | DEPTH=10 (deeper/wider model, same time budget) |
| 7 | a4 | 1.046152 | ok | `8a4b4c1` | DEPTH=6 (shallower/narrower model, same time budget) |
| 11 | a4 | — | crash | `40668d5` | DEPTH=8 baseline, DEVICE_BATCH_SIZE=256 (fewer grad-accum steps, more throughput) |
| 17 | a4 | 0.999520 | ok | `201de54` | DEPTH=8 baseline, WARMDOWN_RATIO=0.3 (shorter LR cooldown, more time at peak LR) |
| 21 | a4 | 0.998686 | ok | `b206781` | DEPTH=8 baseline, WARMDOWN_RATIO=0.7 (longer LR cooldown) |
| 25 | a4 | 0.997158 | ok | `e1c1646` | DEPTH=8 baseline schedule (WARMDOWN=0.5), MATRIX_LR=0.05 (up from 0.04) |

**Best so far: val_bpb 0.997158 by a4 at commit `e1c1646`.**

## Findings (append-only broadcast)

- `#15` **a4** [commit `f461662`]: DEPTH=10 (vs baseline DEPTH=8) gives val_bpb 1.002419, worse than baseline 0.997359. Bigger model trains too slowly within the fixed 5-min budget; fewer optimizer steps outweigh extra capacity.
- `#102` **a4** _(weak claim)_ [commit `201de54`]: WARMDOWN_RATIO=0.3 (vs baseline 0.5) gives val_bpb 0.999520, slightly worse than baseline 0.997359. Shortening LR cooldown does not help here.
- `#113` **a4** _(weak claim)_ [commit `b206781`]: WARMDOWN_RATIO=0.7 gives val_bpb 0.998686, still worse than baseline 0.997359 (WARMDOWN_RATIO=0.5). Combined with the 0.3 result (0.999520), baseline's WARMDOWN_RATIO=0.5 is a local optimum on this axis.
- `#123` **a4** _(weak claim)_ [commit `e1c1646`]: Best result this session: DEPTH=8 baseline architecture/schedule with MATRIX_LR=0.05 (up from 0.04) gives val_bpb 0.997158 vs baseline 0.997359 — a marginal, likely within-noise improvement (single run, no repeat). Depth sweep (6,8,10) and WARMDOWN_RATIO sweep (0.3,0.5,0.7) both confirmed baseline's DEPTH=8/WARMDOWN=0.5 as locally optimal; only the Muon LR bump moved the needle at all, and only slightly.

## Disconfirmations (negative results, attempts to falsify)

- `#89` **a4** [commit `40668d5`]: DEPTH=8 with DEVICE_BATCH_SIZE=256 OOMs (79GB cap, tried to allocate past it). Valid device batch sizes are powers of 2 dividing 256 (TOTAL_BATCH_SIZE/seq_len), so only 128 or 256 are viable at depth 8, and 256 doesn't fit. Batch-size axis exhausted at this depth.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

