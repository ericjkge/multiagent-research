# Shared research directory (open protocol)

_You are `a1`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997333** (commit 228791f)

**Training runs remaining in this cell: 4 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 4 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#3` **a1** — model sizing / depth-width tradeoff: Will tune DEPTH, ASPECT_RATIO, HEAD_DIM, and DEVICE_BATCH_SIZE to find compute-optimal model size for the fixed 5-min budget, keeping optimizer/schedule fixed. Will not assume baseline DEPTH=8 is optimal; will not touch LR schedule shape unless model-size sweep is exhausted.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 1 | a1 | 1.001611 | ok | `9afb04c` | DEPTH=10 (deeper model, ~640 dim) vs baseline DEPTH=8 |
| 8 | a1 | 1.042401 | ok | `88e4480` | DEPTH=6 (shallower/wider, ~384 dim) vs baseline DEPTH=8 |
| 17 | a1 | 0.994598 | ok | `711f50b` | DEPTH=8, ASPECT_RATIO=80 (wider at fixed depth, n_embd=640) |
| 22 | a1 | 1.022835 | ok | `e3cf903` | DEPTH=8, ASPECT_RATIO=96 (n_embd=768, even wider) |
| 27 | a1 | 0.998954 | ok | `f3c084c` | DEPTH=9, ASPECT_RATIO=71 (n_embd=640, same width as best, depth+1) |
| 31 | a1 | 1.005521 | ok | `ea59529` | DEPTH=8, n_embd=640 (best), HEAD_DIM=64 (10 heads instead of 5) |

**Best so far: val_bpb 0.994598 by a1 at commit `711f50b`.**

## Findings (append-only broadcast)

- `#64` **a1** [commit `711f50b`]: DEPTH=8, ASPECT_RATIO=80 (n_embd=640, wider at fixed depth) gives val_bpb 0.994598, BEATS baseline DEPTH=8/ASPECT_RATIO=64 (0.997333) and beats both depth-8-holding-narrower and deeper variants. So at fixed depth=8, widening (512->640) helps net-of-fewer-tokens tradeoff. New best for a1.
- `#89` **a1** [commit `f3c084c`]: DEPTH=9, ASPECT_RATIO=71 (same n_embd=640 as best config, but one layer deeper) gives val_bpb 0.998954, WORSE than DEPTH=8 at the same n_embd=640 (0.994598). Confirms DEPTH=8 is the depth optimum independent of width -- the win at n_embd=640 is specifically about width, not an artifact of depth confounding. Grid so far (n_embd @ depth): 512@8=0.997333(baseline), 640@8=0.994598(best), 640@9=0.998954, 640@10=1.001611, 384@6=1.042401, 768@8=1.022835.

## Disconfirmations (negative results, attempts to falsify)

- `#16` **a1** [commit `9afb04c`]: DEPTH=10 (n_embd=640, deeper/narrower) gives val_bpb 1.001611, WORSE than baseline DEPTH=8 (0.997333). Deeper models don't pay off within the fixed 5-min wall-clock budget at this scale.
- `#40` **a1** [commit `88e4480`]: DEPTH=6 (n_embd=384, shallower/wider) gives val_bpb 1.042401, also WORSE than baseline DEPTH=8 (0.997333), and worse than DEPTH=10 (1.001611). Depth sweep {6,8,10} shows DEPTH=8 is a clear local optimum -- baseline depth/width ratio already well-tuned for this 5-min budget. Moving to width-at-fixed-depth as next lever.
- `#76` **a1** [commit `e3cf903`]: DEPTH=8, ASPECT_RATIO=96 (n_embd=768, the dim AdamW LRs are tuned for) gives val_bpb 1.022835 -- WORSE than both baseline (0.997333) and ASPECT_RATIO=80/n_embd=640 (0.994598). Width sweep at depth=8 -- {512: 0.997333, 640: 0.994598, 768: 1.022835} -- shows a clean local optimum at n_embd=640, not at the LR-tuned 768. Going wider than 640 overshoots badly within the fixed 5-min budget (fewer steps at same wall-clock cost).

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

