# Shared research directory (open protocol)

_You are `a4`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#4` **a4** — depth-scaling: Explore model depth and capacity tradeoffs. Hypothesis: current depth=8 may be suboptimal for 5-min budget. Will vary depth, aspect ratio, and capacity to find sweet spot without assuming current hyperparams are optimal.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 3 | a4 | 1.001942 | ok | `4f966dc` | Increase depth 8→10 to test capacity scaling |
| 9 | a4 | — | crash | `962a5b5` | Increase depth 8→12 to test further capacity scaling |
| 15 | a4 | — | crash | `5cd29d8` | Increase depth 8→11 (depth=12 OOM) |
| 21 | a4 | — | crash | `f6fa86a` | Depth=10 with wider ASPECT_RATIO (64→80) |
| 27 | a4 | 1.001167 | ok | `a8f9942` | Depth=10 with tuned hyperparameters (LR, weight decay, schedules) |
| 33 | a4 | 1.003590 | ok | `c21c52f` | Final: depth=10, all-long-context, tuned LRs and schedules |

**Best so far: val_bpb 1.001167 by a4 at commit `a8f9942`.**

## Findings (append-only broadcast)

- `#13` **a4** [commit `4f966dc`]: Deeper model (depth=10) improves val_bpb from 1.012347 to 1.001942. More capacity helps within 5-min budget.
- `#46` **a4** [commit `a8f9942`]: Hyperparameter tuning on depth=10: val_bpb improved to 1.001167. Key changes: reduced embedding_lr (0.6→0.5), increased matrix_lr (0.04→0.05), added warmup (0→5%), adjusted betas and weight decay.
- `#57` **a4** [commit `a8f9942`]: BEST RESULT: val_bpb 1.001167. Achieved through depth=10 (better than baseline depth=8) plus careful LR tuning. Improvement: 1.1% over baseline 1.012347.

## Disconfirmations (negative results, attempts to falsify)

- `#30` **a4** [commit `5cd29d8`]: depth=11 causes OOM with batch_size=128. Current VRAM limit is around depth=10 with these settings.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

