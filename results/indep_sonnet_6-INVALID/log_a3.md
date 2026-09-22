# Shared research directory (open protocol)

_You are `a3`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997333** (commit 228791f)

**Training runs remaining in this cell: 4 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 4 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#4` **a3** — depth/width scaling + schedule tuning: Will explore increasing model depth/width within the 5-min budget and tuning warmdown/LR to find a better compute-optimal point. Will not assume the baseline DEPTH=8 is optimal; will not touch prepare.py.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 9 | a3 | 1.042821 | ok | `f0bcaa7` | DEPTH 8->6: smaller/faster model, more optimizer steps in fixed 5min budget |

**Best so far: val_bpb 1.042821 by a3 at commit `f0bcaa7`.**

## Findings (append-only broadcast)

- `#19` **a3** [commit `7d32dce`]: DEPTH=8->10 (n_embd 512->640, run crashed/killed mid-training due to session teardown, no val_bpb recorded — but two independent same-family attempts elsewhere in results.tsv (9afb04c val_bpb=1.001611, e2c05e6 val_bpb=1.001200) both show DEPTH=10 is WORSE than baseline 0.997333. Depth increase alone (more params, same time budget => fewer steps) hurts in this 5-min regime.

## Disconfirmations (negative results, attempts to falsify)

_No disconfirmations yet._

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

