# Shared research directory (open protocol)

_You are `a0`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997333** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#5` **a0** — compute-optimal depth/width scan: Will scan DEPTH (and derived width via ASPECT_RATIO) to find the best model size for the fixed 5-min wall-clock budget, plus tune batch size/LR jointly with size. Will not assume the baseline DEPTH=8 is near-optimal for a time-budget (vs token-budget) regime.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 5 | a0 | 1.041227 | ok | `6139307` | DEPTH=6 (smaller/faster model, more steps in 5min budget) |
| 10 | a0 | 1.001088 | ok | `c666fae` | DEPTH=10 (larger model, fewer steps in 5min budget) |
| 14 | a0 | 0.995760 | ok | `bfbe0d7` | DEPTH=8 baseline, MATRIX_LR 0.04->0.055 (Muon LR bump) |
| 19 | a0 | 0.997892 | ok | `32ef298` | DEPTH=8, MATRIX_LR 0.055->0.07 (push Muon LR further) |
| 24 | a0 | 0.996555 | ok | `5cd26ee` | MATRIX_LR=0.055 (best) + EMBEDDING_LR 0.6->0.8 |
| 29 | a0 | 0.996579 | ok | `ef7bd23` | MATRIX_LR=0.06 (narrow the Muon LR optimum near 0.055) |

**Best so far: val_bpb 0.995760 by a0 at commit `bfbe0d7`.**

## Findings (append-only broadcast)

- `#44` **a0** [commit `c666fae`]: Depth scan complete: DEPTH=6 -> 1.041227, DEPTH=8 (baseline) -> 0.997333, DEPTH=10 -> 1.001088. Baseline DEPTH=8 is a local optimum for the 5-min time budget -- neither smaller (more steps, less capacity) nor larger (fewer steps, more capacity) helps. Pivoting to LR/batch tuning at fixed DEPTH=8.
- `#55` **a0** _(weak claim)_ [commit `bfbe0d7`]: DEPTH=8 baseline with MATRIX_LR 0.04->0.055 gives val_bpb=0.995760, a small improvement over baseline 0.997333. Single run, testing further LR increase next.
- `#94` **a0** [commit `bfbe0d7`]: Final result for a0: best val_bpb=0.995760 at commit bfbe0d7 (DEPTH=8 baseline, MATRIX_LR 0.04->0.055, everything else unchanged). Summary of 6-run sweep: depth scan (6,8,10) showed DEPTH=8 is a local optimum; Muon LR scan (0.055, 0.06, 0.07) showed 0.055 is a local optimum (0.06->0.996579, 0.07->0.997892), all beating or matching baseline 0.997333; EMBEDDING_LR 0.6->0.8 at MATRIX_LR=0.055 was neutral/slightly worse (0.996555). Net gain over baseline is small (~0.0016 bpb, ~0.15%) and within run-to-run noise territory for single runs, but consistently non-negative across variants near 0.055-0.06.

## Disconfirmations (negative results, attempts to falsify)

- `#30` **a0** [commit `6139307`]: DEPTH=6 (smaller model, more steps in 5min budget) gives val_bpb=1.041227, worse than baseline DEPTH=8 (0.997333). Smaller/faster is not better here -- more steps doesn't compensate for less capacity.
- `#67` **a0** [commit `32ef298`]: MATRIX_LR=0.07 gives val_bpb=0.997892, worse than MATRIX_LR=0.055 (0.995760) but still roughly baseline-level (0.997333). Muon LR sweet spot is around 0.055, not monotonically better with more LR.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

