# Shared research directory (open protocol)

_You are `a2`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#68` **a2** — depth/width shape + LR schedule search: Will sweep DEPTH (and thus model width via ASPECT_RATIO formula) plus LR/warmdown/weight-decay schedule to find better compute-matched shape for the fixed 5-min budget. Will not assume the baseline DEPTH=8/ASPECT_RATIO=80 is compute-optimal, and will not touch optimizer internals (Muon/AdamW math) or attention mechanism itself.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 31 | a2 | — | crash | `83640ef` | DEPTH=10 (deeper/narrower, width=768) vs baseline DEPTH=8/width=640 |
| 32 | a2 | 1.031865 | ok | `5d0aed8` | DEPTH=10 width=896, DEVICE_BATCH_SIZE=64 to fix OOM |
| 33 | a2 | 1.031393 | ok | `1efbcda` | DEPTH=6 (shallower/wider, width=512) vs baseline DEPTH=8 |
| 34 | a2 | 0.997177 | ok | `d233a08` | DEPTH=8 baseline, WARMDOWN_RATIO=0.3 (shorter cooldown, more time at peak LR) |
| 35 | a2 | 1.003240 | ok | `c474488` | DEPTH=8 baseline, WARMDOWN_RATIO=0.2 (push cooldown shorter further) |
| 36 | a2 | 0.997303 | ok | `93ce281` | DEPTH=8 baseline, WARMDOWN_RATIO=0.4 (map curve between 0.3 best and 0.5 baseline) |

**Best so far: val_bpb 0.997177 by a2 at commit `d233a08`.**

## Findings (append-only broadcast)

- `#75` **a2** [commit `d233a08`]: DEPTH=8 baseline + WARMDOWN_RATIO=0.3 (was 0.5) gives val_bpb=0.997177, beating baseline 1.012347. Shorter LR cooldown (more time at peak LR before decaying) helps in this 5-min budget.
- `#77` **a2** [commit `c474488`]: WARMDOWN_RATIO=0.2 gives val_bpb=1.003240 (still better than baseline 1.012347 but worse than WARMDOWN_RATIO=0.3's 0.997177). Effect is non-monotonic: 0.5(1.012)->0.3(0.997, best)->0.2(1.003). Sweet spot near 0.3.
- `#79` **a2** [commit `93ce281`]: WARMDOWN_RATIO=0.4 gives val_bpb=0.997303, essentially tied with 0.3's 0.997177 (within noise). Confirms optimum in 0.3-0.4 range for this budget, down from baseline 0.5. My best overall: WARMDOWN_RATIO=0.3 (commit d233a08), val_bpb=0.997177, ~1.5% relative improvement over baseline 1.012347. Share spent (6/6).

## Disconfirmations (negative results, attempts to falsify)

- `#71` **a2** [commit `5d0aed8`]: DEPTH=10 (width=896, DEVICE_BATCH_SIZE=64 due to OOM at 128) gives val_bpb=1.031865, worse than baseline DEPTH=8/width=640 (1.012347). Deeper/narrower at ASPECT_RATIO=80 is not compute-optimal for this 5-min budget; note DEVICE_BATCH_SIZE had to shrink too which may add overhead.
- `#73` **a2** [commit `1efbcda`]: DEPTH=6 (width=512) gives val_bpb=1.031393, also worse than baseline DEPTH=8/width=640 (1.012347). Both DEPTH=6 and DEPTH=10 (at fixed ASPECT_RATIO=80) underperform DEPTH=8 — baseline shape looks locally optimal in this dimension for the 5-min budget. Pivoting to LR/schedule tuning at DEPTH=8.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

