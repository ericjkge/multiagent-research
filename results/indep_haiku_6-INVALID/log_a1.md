# Shared research directory (open protocol)

_You are `a1`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997359** (commit 228791f)

**Training runs remaining in this cell: 23 of 36.**
Each agent's share is 6 runs. Used so far: a0: 5 left, a1: 3 left, a2: 5 left, a4: 4 left, a5: 0 left.

## Approaches (slots)

- `#4` **a1** — hyperparameter-tuning: Systematic exploration of learning rates, weight decay schedules, and model architecture configuration to optimize val_bpb within the 5-minute budget.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 3 | a1 | 0.998188 | ok | `ac0d45e` | Reduce embedding LR from 0.6 to 0.4 |
| 12 | a1 | 0.996838 | ok | `0d7bf5c` | Increase MATRIX_LR from 0.04 to 0.08 |

**Best so far: val_bpb 0.996838 by a1 at commit `0d7bf5c`.**

## Findings (append-only broadcast)

- `#25` **a1** [commit `ac0d45e`]: First attempt (EMBEDDING_LR=0.4) resulted in 0.998188, worse than baseline 0.997359. Learned that embedding LR reduction helps overfitting but hurts overall performance.

## Disconfirmations (negative results, attempts to falsify)

_No disconfirmations yet._

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

