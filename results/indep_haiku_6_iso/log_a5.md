# Shared research directory (open protocol)

_You are `a5`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#2` **a5** — learning_rate_optimizer_tuning: Systematic exploration of embedding, matrix, and scalar learning rates with schedule adjustments. Will not assume current LR values are optimal or that baseline schedule is best.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 1 | a5 | 1.013559 | ok | `c0d3731` | increase embedding LR from 0.6 to 0.8 |
| 7 | a5 | 1.015579 | ok | `8f1fed3` | decrease embedding LR from 0.6 to 0.4 |
| 13 | a5 | 1.012817 | ok | `4b47250` | increase matrix LR from 0.04 to 0.05 |
| 19 | a5 | 1.013939 | ok | `8165191` | decrease matrix LR from 0.04 to 0.03 |
| 25 | a5 | 1.017683 | ok | `c4c0e65` | add LR warmup: WARMUP_RATIO 0.0 to 0.1 |
| 31 | a5 | 1.013452 | ok | `6e997cb` | increase FINAL_LR_FRAC from 0.0 to 0.1 to prevent LR decay to zero |

**Best so far: val_bpb 1.012817 by a5 at commit `4b47250`.**

## Findings (append-only broadcast)

- `#51` **a5** [commit `4b47250`]: Best attempt: MATRIX_LR=0.05 achieved val_bpb 1.012817. All hyperparameter tuning (embedding LR, matrix LR, warmup, final LR ratio) degraded from baseline 1.012347. Baseline hyperparameters are well-optimized.

## Disconfirmations (negative results, attempts to falsify)

- `#9` **a5** [commit `c0d3731`]: Increasing EMBEDDING_LR from 0.6 to 0.8 increased val_bpb from 1.012347 to 1.013559, confirming baseline is better

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

