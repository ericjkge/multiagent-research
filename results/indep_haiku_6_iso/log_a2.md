# Shared research directory (open protocol)

_You are `a2`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#3` **a2** — LR-and-depth-search: Systematic exploration of learning rates, model depth, and batch configurations. Will not assume current hyperparameters are optimal.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 2 | a2 | 1.001925 | ok | `d454df7` | Increase depth from 8 to 10 layers |
| 8 | a2 | — | crash | `d9e9b68` | Increase depth to 12 layers |
| 14 | a2 | 1.004821 | ok | `f86de1c` | Depth 10 + increase MATRIX_LR from 0.04 to 0.06 |
| 20 | a2 | 0.999826 | ok | `62e2ffb` | Depth 10 + increase EMBEDDING_LR from 0.6 to 0.7 |
| 26 | a2 | 1.002127 | ok | `080a91d` | Depth 10 + EMBEDDING_LR 0.8 |
| 32 | a2 | 1.001223 | ok | `ff7a446` | Depth 10 + EMBEDDING_LR 0.75 (final run) |

**Best so far: val_bpb 0.999826 by a2 at commit `62e2ffb`.**

## Findings (append-only broadcast)

- `#11` **a2** [commit `d454df7`]: Increasing depth to 10 layers improves val_bpb to 1.001925 (was 1.012347). Deeper models better utilize the 5-minute budget.
- `#37` **a2** [commit `62e2ffb`]: Increasing EMBEDDING_LR from 0.6 to 0.7 with depth 10 achieves 0.999826. Embedding learning rate is important for this model.
- `#54` **a2** [commit `62e2ffb`]: Final best result: val_bpb 0.999826 (1.2% improvement over baseline). Achieved with DEPTH=10, EMBEDDING_LR=0.7, other hyperparameters default.

## Disconfirmations (negative results, attempts to falsify)

- `#21` **a2** [commit `d9e9b68`]: Depth 12 causes CUDA OOM with 79GB GPU. Optimal appears to be around depth 10.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

