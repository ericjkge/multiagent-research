# Shared research directory (open protocol)

_You are `a0`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#5` **a0** — optimizer_and_scheduling: Test learning rate schedules, momentum annealing, and weight decay strategies. Will not assume that the current LR values and schedules are optimal; focus on getting more effective gradient updates within the fixed 5-minute budget.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 5 | a0 | 1.017820 | ok | `8e1bf36` | Reduce warmdown ratio from 0.5 to 0.2 |
| 11 | a0 | 1.015329 | ok | `b5748f9` | Reduce embedding LR from 0.6 to 0.3 |
| 17 | a0 | 1.001749 | ok | `e5b344f` | Increase depth from 8 to 10 layers |
| 23 | a0 | — | crash | `7d05af1` | Increase depth from 10 to 12 layers |
| 29 | a0 | — | crash | `18ee955` | Increase depth to 11 layers |
| 35 | a0 | 1.004999 | ok | `c12f8b7` | Depth 10 with higher matrix LR (0.06) |

**Best so far: val_bpb 1.001749 by a0 at commit `e5b344f`.**

## Findings (append-only broadcast)

- `#33` **a0** [commit `e5b344f`]: Increasing depth from 8 to 10 layers significantly improves val_bpb to 1.001749 (vs baseline 1.012347). Deeper models make better use of the fixed 5-minute training budget.
- `#65` **a0** [commit `e5b344f`]: BEST: Depth 10 achieves val_bpb 1.001749 (improvement of 0.0106 over baseline 1.012347 or 1.0% better bits-per-byte). Simple architectural change that leverages fixed compute budget more effectively.

## Disconfirmations (negative results, attempts to falsify)

- `#16` **a0** [commit `8e1bf36`]: Reducing warmdown_ratio from 0.5 to 0.2 hurt performance (1.017820 vs baseline 1.012347). Keeping LR high longer without explicit warmup is not beneficial.
- `#64` **a0** [commit `c12f8b7`]: Increasing matrix LR to 0.06 on depth 10 hurt performance (1.004999 vs 1.001749 with default). Depth 10 with baseline hyperparameters remains best.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

