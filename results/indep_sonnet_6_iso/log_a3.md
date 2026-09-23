# Shared research directory (open protocol)

_You are `a3`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#4` **a3** — depth/width scaling for fixed time budget: I will sweep DEPTH (and derived width via ASPECT_RATIO/HEAD_DIM) to find the compute-optimal model size for a fixed 5-minute wall-clock budget, since the baseline depth=8 was not necessarily tuned for a time-limited (vs token-limited) regime. I will not assume LR/optimizer hyperparams are already optimal for the new sizes and will do a light LR re-check if depth changes a lot.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 4 | a3 | 1.066017 | ok | `c0f2fbb` | DEPTH=6 (smaller/faster model, more steps in 5min budget) |
| 9 | a3 | — | crash | `6f0db0a` | DEPTH=12 (larger model, fewer steps in 5min budget) |
| 14 | a3 | 1.015028 | ok | `07b25a7` | DEPTH=12, DEVICE_BATCH_SIZE=64 (avoid OOM, larger model fewer steps) |
| 19 | a3 | 1.011362 | ok | `f3832e5` | DEPTH=8 baseline, MATRIX_LR 0.04->0.06 (higher Muon LR) |
| 24 | a3 | 1.013105 | ok | `82ca31e` | DEPTH=8 baseline, MATRIX_LR 0.04->0.08 (push Muon LR further) |
| 29 | a3 | 1.012279 | ok | `5a1ab5d` | DEPTH=8, MATRIX_LR=0.06 (repeat, confirm reproducibility of best result) |

**Best so far: val_bpb 1.011362 by a3 at commit `f3832e5`.**

## Findings (append-only broadcast)

- `#13` **a3** [commit `c0f2fbb`]: DEPTH=6 (smaller model, more steps/sec) gives val_bpb 1.066017, worse than baseline 1.012347 (DEPTH=8). Under this 5-min time budget, shrinking depth hurts -- extra steps don't compensate for reduced capacity.
- `#32` **a3** [commit `07b25a7`]: DEPTH=12 (DEVICE_BATCH_SIZE=64 to avoid OOM) gives val_bpb 1.015028, essentially tied with baseline 1.012347 (DEPTH=8) -- not an improvement. Combined with DEPTH=6 (1.066017, clearly worse), DEPTH=8 looks close to the compute-optimal depth for this 5min budget; depth sweep alone won't beat baseline.
- `#38` **a3** _(weak claim)_ [commit `f3832e5`]: DEPTH=8, MATRIX_LR 0.04->0.06: val_bpb 1.011362 vs baseline 1.012347. Small improvement, single run. Trying 0.08 next to see if trend continues.
- `#61` **a3** [commit `5a1ab5d`]: Repeat of DEPTH=8, MATRIX_LR=0.06: val_bpb 1.012279, vs first run 1.011362 and baseline 1.012347. The two 0.06 runs differ by ~0.001, same order as the apparent improvement over baseline -- this gain is within run-to-run noise, not a robust win. Overall verdict: depth sweep (6,8,12) and matrix_lr sweep (0.04,0.06,0.08) both show baseline DEPTH=8/MATRIX_LR=0.04 is already close to a local optimum for this 5min budget; no configuration I tried clearly beat it.

## Disconfirmations (negative results, attempts to falsify)

- `#20` **a3** [commit `6f0db0a`]: DEPTH=12 with DEVICE_BATCH_SIZE=128 OOMs (79GB GPU). Retrying with smaller device batch size.
- `#45` **a3** [commit `82ca31e`]: DEPTH=8, MATRIX_LR=0.08: val_bpb 1.013105, worse than MATRIX_LR=0.06 (1.011362) and about tied with baseline 0.04 (1.012347). LR-vs-bpb is non-monotonic; 0.06 looks like a local sweet spot, not 'higher is better'.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

