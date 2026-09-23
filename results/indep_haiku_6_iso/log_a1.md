# Shared research directory (open protocol)

_You are `a1`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#6` **a1** — optimizer-tuning: Test learning rate and optimizer hyperparameters systematically. Will not assume current LR settings are optimal for the 5-minute training window; focusing on embedding_lr, matrix_lr, and warmup schedule.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 4 | a1 | 1.012457 | ok | `384bc42` | Reduce EMBEDDING_LR from 0.6 to 0.3 |
| 10 | a1 | 1.013510 | ok | `f8d62ee` | Increase MATRIX_LR from 0.04 to 0.08 |
| 16 | a1 | 1.014915 | ok | `47fe7ca` | Add warmup_ratio 0.05 for training stability |
| 22 | a1 | 1.001943 | ok | `012fd8c` | Increase DEPTH from 8 to 10 |
| 28 | a1 | — | crash | `16e7e31` | Increase DEPTH from 10 to 12 |
| 34 | a1 | — | crash | `f05aa67` | Increase DEPTH from 10 to 11 |

**Best so far: val_bpb 1.001943 by a1 at commit `012fd8c`.**

## Findings (append-only broadcast)

- `#40` **a1** [commit `012fd8c`]: Increasing depth from 8 to 10 significantly improves val_bpb to 1.001943, well below baseline of 1.012347. Depth increase allows better model capacity utilization within the 5-minute training window.
- `#60` **a1** [commit `012fd8c`]: Best result: val_bpb=1.001943 achieved with DEPTH=10. This represents a 1.0% improvement over baseline (1.012347). Increasing depth from 8 to 10 provides sufficient additional model capacity to better utilize the 5-minute training budget without exceeding VRAM constraints (68GB peak).

## Disconfirmations (negative results, attempts to falsify)

- `#59` **a1** [commit `16e7e31`]: Depth=12 causes CUDA OOM with peak VRAM 78.4GB in use. Model too large for single H100.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

