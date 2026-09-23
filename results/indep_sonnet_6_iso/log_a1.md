# Shared research directory (open protocol)

_You are `a1`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#3` **a1** — depth/width & LR scan: Will scan DEPTH (model depth vs width under fixed ASPECT_RATIO) and Muon/Adam LR scaling around the baseline, holding architecture (value embeddings, sliding window, NorMuon) fixed. Will not touch the window pattern or optimizer internals.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 2 | a1 | 1.001799 | ok | `91b61c0` | DEPTH=10 (deeper/narrower under fixed ASPECT_RATIO) |
| 7 | a1 | — | crash | `9329028` | DEPTH=12 (push deeper further) |
| 12 | a1 | 1.015012 | ok | `d06309a` | DEPTH=12, DEVICE_BATCH_SIZE=64 to fix OOM |
| 17 | a1 | 1.004844 | ok | `e883f82` | DEPTH=10, MATRIX_LR=0.05 (Muon LR bump) |
| 22 | a1 | 1.002009 | ok | `da43376` | DEPTH=10 (best config), WARMDOWN_RATIO=0.35 (shorter cooldown) |
| 27 | a1 | 1.002941 | ok | `984f2f7` | DEPTH=10 (best config), WEIGHT_DECAY=0.1 (final run of share) |

**Best so far: val_bpb 1.001799 by a1 at commit `91b61c0`.**

## Findings (append-only broadcast)

- `#10` **a1** [commit `91b61c0`]: DEPTH=10 (vs baseline DEPTH=8) improves val_bpb 1.012347 -> 1.001799 under fixed 5min budget. Deeper/narrower model wins despite fewer steps.
- `#52` **a1** [commit `91b61c0`]: Final best for a1: DEPTH=10 with otherwise-default hyperparams, val_bpb=1.001799 (vs baseline 1.012347). Tried DEPTH=12 (OOM at batch=128; with batch=64 got worse, 1.015012 — depth=10 is a local sweet spot), MATRIX_LR=0.05 (1.004844, worse), WARMDOWN_RATIO=0.35 (1.002009, ~tied/worse), WEIGHT_DECAY=0.1 (1.002941, worse). None beat plain DEPTH=10.

## Disconfirmations (negative results, attempts to falsify)

- `#26` **a1** [commit `d06309a`]: DEPTH=12 with DEVICE_BATCH_SIZE=64 (halved to avoid OOM) gives val_bpb 1.015012, worse than both baseline (1.012347) and DEPTH=10 (1.001799). Depth=10 appears near the sweet spot for this 5min budget; going deeper costs too much per-step throughput and grad_accum overhead.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

