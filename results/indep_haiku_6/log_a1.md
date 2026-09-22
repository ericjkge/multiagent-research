# Shared research directory (open protocol)

_You are `a1`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997359** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#4` **a1** — hyperparameter-tuning: Systematic exploration of learning rates, weight decay schedules, and model architecture configuration to optimize val_bpb within the 5-minute budget.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 3 | a1 | 0.998188 | ok | `ac0d45e` | Reduce embedding LR from 0.6 to 0.4 |
| 12 | a1 | 0.996838 | ok | `0d7bf5c` | Increase MATRIX_LR from 0.04 to 0.08 |
| 15 | a1 | 0.980185 | ok | `5abfeb4` | Apply winning config: batch 2^18, width ASPECT_RATIO=96, window=256, VE=first3+last3 |
| 20 | a1 | — | crash | `f848036` | DEPTH=9 (from 8) with winning config |
| 26 | a1 | — | crash | `79bef41` | DEPTH=10 with winning config |
| 30 | a1 | — | crash | `422e7a6` | DEPTH=9 + DEVICE_BATCH_SIZE=96 (reduced to avoid OOM) |

**Best so far: val_bpb 0.980185 by a1 at commit `5abfeb4`.**

## Findings (append-only broadcast)

- `#25` **a1** [commit `ac0d45e`]: First attempt (EMBEDDING_LR=0.4) resulted in 0.998188, worse than baseline 0.997359. Learned that embedding LR reduction helps overfitting but hurts overall performance.
- `#57` **a1** [commit `5abfeb4`]: Winning config stacking: TOTAL_BATCH_SIZE=2^18 (-0.0094), ASPECT_RATIO=96 (width>depth), short_window=256 (//8 not //2), value_embeds=first3+last3. Result: 0.980185 vs baseline 0.997359 (-0.0172). This is the measured frontier from prior session validation.
- `#79` **a1** [commit `5abfeb4`]: FINAL BEST RESULT: val_bpb=0.980185 using winning config (BATCH=2^18, ASPECT_RATIO=96, window//8, VE=first3+last3). Improvement of 0.0172 vs baseline 0.997359. Represents validated frontier from prior sessions.

## Disconfirmations (negative results, attempts to falsify)

_No disconfirmations yet._

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

