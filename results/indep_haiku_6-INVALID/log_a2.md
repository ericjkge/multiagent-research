# Shared research directory (open protocol)

_You are `a2`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997359** (commit 228791f)

**Training runs remaining in this cell: 23 of 36.**
Each agent's share is 6 runs. Used so far: a0: 5 left, a1: 3 left, a2: 5 left, a4: 4 left, a5: 0 left.

## Approaches (slots)

- `#6` **a2** — depth-width-lr-tradeoffs: Explore depth vs width tradeoffs and learning rate schedules to minimize val_bpb in 5-minute training. Will not assume current hyperparameters are near-optimal; will test systematic architecture and optimizer variations.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 5 | a2 | 1.002563 | ok | `a0ac46e` | Increase depth from 8 to 10, reduce aspect ratio 64→52 |

**Best so far: val_bpb 1.002563 by a2 at commit `a0ac46e`.**

## Findings (append-only broadcast)

- `#37` **a2** [commit `a0ac46e`]: Negative result: Depth increase (8→10 layers, aspect_ratio 64→52) achieved val_bpb 1.002563, worse than baseline 0.997359. Despite keeping total parameters similar via aspect ratio reduction, deeper models underperform in 5-minute budget. Suggests diminishing returns to depth scaling in short-horizon training.

## Disconfirmations (negative results, attempts to falsify)

- `#31` **a2** [commit `a0ac46e`]: Increasing depth from 8 to 10 while reducing aspect ratio 64→52 hurt performance: val_bpb 1.002563 vs baseline 0.997359. Deeper networks at this scale don't help in 5-minute budget.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

