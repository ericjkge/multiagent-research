# Shared research directory (open protocol)

_You are `a4`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#6` **a4** — compute-optimal depth/width under fixed 5-min wall-clock: I will sweep DEPTH (and the derived model_dim/n_head via ASPECT_RATIO/HEAD_DIM) to find the depth that maximizes tokens-processed-quality tradeoff within the fixed time budget, plus complementary LR/schedule tuning. I will not assume the current DEPTH=8 default is optimal for this fixed wall-clock budget, since deeper models get fewer steps but may be more sample-efficient.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 3 | a4 | 1.066267 | ok | `2149a5f` | shallower DEPTH=6 (vs default 8) for more steps in fixed time budget |
| 8 | a4 | — | crash | `52d5bd7` | deeper DEPTH=12 (vs default 8) |
| 13 | a4 | 1.014859 | ok | `7b6369f` | deeper DEPTH=12 with DEVICE_BATCH_SIZE=64 to avoid OOM |
| 18 | a4 | 1.012919 | ok | `78696ea` | DEPTH=8 baseline with MATRIX_LR=0.055 (up from 0.04) |
| 23 | a4 | 1.013932 | ok | `2c2ef38` | DEPTH=8 baseline with WARMDOWN_RATIO=0.35 (shorter cooldown, was 0.5) |
| 28 | a4 | 1.014791 | ok | `8c164c1` | DEPTH=8 baseline with WINDOW_PATTERN=SSLL (more full-context layers, was SSSL) |

**Best so far: val_bpb 1.012919 by a4 at commit `78696ea`.**

## Findings (append-only broadcast)

- `#28` **a4** [commit `2149a5f`]: DEPTH=6 (shallower/wider, more steps): val_bpb=1.066267, worse than baseline 1.012347 (DEPTH=8).
- `#29` **a4** [commit `7b6369f`]: DEPTH=12 (deeper, DEVICE_BATCH_SIZE=64 to avoid OOM): val_bpb=1.014859, roughly on par but slightly worse than baseline DEPTH=8 (1.012347). Depth=8 default looks near-optimal for this fixed 5-min budget; deviating either direction hurts.
- `#55` **a4** [commit `78696ea`]: DEPTH=8 with MATRIX_LR=0.055: val_bpb=1.012919, essentially tied with baseline 1.012347 (within noise), best of my 6 runs.
- `#56` **a4** _(weak claim)_ [commit `2c2ef38`]: WARMDOWN_RATIO=0.35 (shorter cooldown): val_bpb=1.013932, slightly worse than baseline, single run.
- `#57` **a4** _(weak claim)_ [commit `8c164c1`]: WINDOW_PATTERN=SSLL (more full-context layers): val_bpb=1.014791, slightly worse than baseline SSSL pattern, single run.

## Disconfirmations (negative results, attempts to falsify)

- `#30` **a4** [commit `2149a5f`]: Shallower DEPTH=6 is clearly worse (1.066 vs 1.012), disconfirming the idea that more optimizer steps from a smaller model beats the default depth within the fixed time budget.
- `#58` **a4** [commit `52d5bd7`]: Across depth (6,8,12), matrix_lr, warmdown_ratio, and window_pattern sweeps, nothing beat the untouched baseline (DEPTH=8, val_bpb=1.012347) within this 5-min budget; the given hyperparameters appear well-tuned already for this compute-optimal-depth family.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

