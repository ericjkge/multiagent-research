# Shared research directory (open protocol)

Baseline at start of the cell: **val_bpb 0.996598** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 36 runs. Used so far: a0: 0 left.

## Approaches (slots)

- `#14` **a0** — optimizer_tuning: Systematic optimization of learning rates, weight decay, and Adam betas to improve convergence speed and final validation loss. Will not assume current default hyperparameters are optimal.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 1 | a0 | 0.996841 | ok | `228791f` | Baseline: unchanged code, establish reproducibility |
| 2 | a0 | 0.997250 | ok | `23a6dcb` | Increase EMBEDDING_LR from 0.6 to 0.8 |
| 3 | a0 | 0.996279 | ok | `3a8b696` | Increase MATRIX_LR from 0.04 to 0.05 |
| 4 | a0 | 0.996071 | ok | `a746487` | Increase MATRIX_LR from 0.05 to 0.06 |
| 5 | a0 | 0.997070 | ok | `6c14e52` | Increase MATRIX_LR to 0.07 |
| 6 | a0 | 0.995559 | ok | `0670d9e` | MATRIX_LR=0.06 + WEIGHT_DECAY=0.15 |
| 7 | a0 | 0.995570 | ok | `50011d5` | MATRIX_LR=0.06 + WEIGHT_DECAY=0.1 |
| 8 | a0 | 0.995551 | ok | `8322efc` | MATRIX_LR=0.06 + WEIGHT_DECAY=0.15 + UNEMBEDDING_LR=0.006 |
| 9 | a0 | 0.994809 | ok | `0948405` | SCALAR_LR reduced from 0.5 to 0.4 |
| 10 | a0 | 0.995800 | ok | `b2c4591` | SCALAR_LR reduced to 0.3 |
| 11 | a0 | 0.995158 | ok | `2c67985` | EMBEDDING_LR=0.65, SCALAR_LR=0.4 |
| 12 | a0 | 0.995398 | ok | `fbd84e4` | MATRIX_LR=0.055, SCALAR_LR=0.4 |
| 13 | a0 | 0.995942 | ok | `0b41b15` | WEIGHT_DECAY=0.12, keeping MATRIX_LR=0.06, SCALAR_LR=0.4 |
| 14 | a0 | 0.995055 | ok | `9b8b86e` | ADAM_BETAS=(0.75, 0.95) |
| 15 | a0 | 0.996245 | ok | `87ac330` | ADAM_BETAS=(0.85, 0.95) |
| 16 | a0 | 0.994955 | ok | `b99cc98` | ADAM_BETAS=(0.8, 0.92) |
| 17 | a0 | 0.994780 | ok | `3972452` | ADAM_BETAS=(0.8, 0.90) |
| 18 | a0 | 0.994851 | ok | `027daf6` | ADAM_BETAS=(0.8, 0.88) |
| 19 | a0 | 0.999977 | ok | `0a331ca` | DEPTH=10 with best optimizer hyperparams |
| 20 | a0 | 1.043521 | ok | `78ae3ad` | DEPTH=6 with best optimizer hyperparams |
| 21 | a0 | 0.994182 | ok | `5d48413` | ASPECT_RATIO=70 with best optimizer hyperparams |
| 22 | a0 | 0.994092 | ok | `a0168ca` | ASPECT_RATIO=75 |
| 23 | a0 | 0.994104 | ok | `2fd23b8` | ASPECT_RATIO=80 |
| 24 | a0 | 0.994877 | ok | `7511cd2` | ASPECT_RATIO=75 + WARMDOWN_RATIO=0.4 |
| 25 | a0 | 0.993136 | ok | `94417e6` | ASPECT_RATIO=75 + EMBEDDING_LR=0.7 |
| 26 | a0 | 0.993211 | ok | `1a05ebd` | ASPECT_RATIO=75 + EMBEDDING_LR=0.75 |
| 27 | a0 | 0.993672 | ok | `fca4ce4` | ASPECT_RATIO=75 + EMBEDDING_LR=0.7 + MATRIX_LR=0.065 |
| 28 | a0 | 0.995876 | ok | `7c50c77` | ASPECT_RATIO=75 + EMBEDDING_LR=0.7 + SCALAR_LR=0.45 |
| 29 | a0 | 0.993737 | ok | `722711d` | ASPECT_RATIO=75 + EMBEDDING_LR=0.7 + WEIGHT_DECAY=0.12 |
| 30 | a0 | 0.993469 | ok | `217a664` | ASPECT_RATIO=73 + EMBEDDING_LR=0.7 |
| 31 | a0 | 0.992818 | ok | `d288325` | ASPECT_RATIO=77 + EMBEDDING_LR=0.7 |
| 32 | a0 | 0.993146 | ok | `652779a` | ASPECT_RATIO=78 + EMBEDDING_LR=0.7 |
| 33 | a0 | 0.993896 | ok | `6eea013` | ASPECT_RATIO=77 + EMBEDDING_LR=0.68 |
| 34 | a0 | 0.995399 | ok | `a30561f` | ASPECT_RATIO=77 + EMBEDDING_LR=0.72 |
| 35 | a0 | 0.995511 | ok | `96f42c2` | ASPECT_RATIO=77 + EMBEDDING_LR=0.7 + MATRIX_LR=0.064 |
| 36 | a0 | 0.994692 | ok | `0718f6f` | ASPECT_RATIO=77 + EMBEDDING_LR=0.7 + WINDOW_PATTERN=SLSL |

**Best so far: val_bpb 0.992818 by a0 at commit `d288325`.**

## Findings (append-only broadcast)

- `#33` **a0** [commit `3972452`]: ADAM_BETAS=(0.8, 0.90) with MATRIX_LR=0.06, SCALAR_LR=0.4, WEIGHT_DECAY=0.15, UNEMBEDDING_LR=0.006 achieves val_bpb 0.994780, a 0.18% improvement over baseline 0.996841
- `#45` **a0** [commit `94417e6`]: ASPECT_RATIO=75 + EMBEDDING_LR=0.7 with optimal optimizer params achieves val_bpb 0.993136, a 0.36% improvement over baseline
- `#53` **a0** [commit `d288325`]: FINAL BEST: ASPECT_RATIO=77 + EMBEDDING_LR=0.7 + optimal optimizer params achieves val_bpb 0.992818, a 0.38% improvement over baseline 0.996841

## Disconfirmations (negative results, attempts to falsify)

_No disconfirmations yet._

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

