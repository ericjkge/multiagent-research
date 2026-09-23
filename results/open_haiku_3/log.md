# Shared research directory (open protocol)

Baseline at start of the cell: **val_bpb 0.996598** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 12 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left.

## Approaches (slots)

- `#2` **a1** — model_architecture: Exploring depth/width tradeoffs and model size optimization. Will test different DEPTH and ASPECT_RATIO combinations to find the best val_bpb within the 5-minute budget constraint, without assuming current dimensions are optimal.
- `#3` **a0** — architecture: Exploring model depth vs width tradeoffs and architectural variations (activation functions, window patterns, attention mechanisms). Will not assume current defaults are optimal; will focus on inference-efficient changes that fit in 5-minute training window.
- `#4` **a2** — optimizer: Systematic tuning of learning rates, weight decay, and optimizer momentum. Will not assume current hyperparameters are optimal; exploring LR scales across parameter groups and weight decay strategies.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 1 | a2 | 0.997355 | ok | `3984c16` | Reduce EMBEDDING_LR from 0.6 to 0.3 |
| 2 | a1 | 1.000596 | ok | `7e083dc` | Increase depth from 8 to 10 |
| 3 | a0 | 1.000690 | ok | `f690b02` | Increased depth (10 layers), reduced width (640 dim) - test deeper/narrower |
| 4 | a2 | 0.997786 | ok | `716bfff` | Increase MATRIX_LR from 0.04 to 0.08 |
| 5 | a0 | 0.991949 | ok | `3ee9f66` | Reduced depth (6 layers), full width (768 dim) - test shallower |
| 6 | a1 | 0.998305 | ok | `e27228b` | Reduce depth to 6, increase ASPECT_RATIO to 80 (shallower, wider) |
| 7 | a2 | 0.997963 | ok | `f5e0390` | EMBEDDING_LR=0.5 (midway tune) |
| 8 | a1 | 0.998266 | ok | `44dd81f` | DEPTH=8 with ASPECT_RATIO=72 (slightly wider baseline) |
| 9 | a1 | 0.998600 | ok | `7be1515` | Reduce WARMDOWN_RATIO from 0.5 to 0.3 (keep higher learning rate longer) |
| 10 | a1 | 0.999405 | ok | `ea47532` | Change MLP activation from relu(x).square() to gelu(x) |
| 11 | a1 | 0.998181 | ok | `2b800c4` | Try narrower model with ASPECT_RATIO=56 |
| 12 | a1 | 0.998891 | ok | `3078ea7` | Increase EMBEDDING_LR from 0.6 to 0.8 |
| 13 | a1 | 0.998386 | ok | `c04e5c1` | Change WINDOW_PATTERN from SSSL to LLLL (all full attention) |
| 14 | a1 | 0.994809 | ok | `82ce95e` | Reduce HEAD_DIM from 128 to 96 |
| 15 | a1 | 0.998048 | ok | `91dc2df` | Increase HEAD_DIM from 128 to 160 |
| 16 | a1 | 0.998169 | ok | `a1f96ff` | Adopt a2's EMBEDDING_LR=0.3 (best result so far: 0.997355) |
| 17 | a1 | 0.998012 | ok | `8d8095c` | EMBEDDING_LR=0.3 + WEIGHT_DECAY=0.1 (reduced weight decay) |
| 18 | a0 | 1.039023 | ok | `bd02971` | Further reduced depth (5 layers), full width (768 dim) |
| 19 | a2 | 0.996691 | ok | `0153e86` | Reduce WEIGHT_DECAY from 0.2 to 0.1 |
| 20 | a2 | 0.996702 | ok | `a2ff105` | Adopt a0's shallower/wider model (DEPTH=6, ASPECT_RATIO=128) |
| 21 | a0 | 1.037945 | ok | `60031ee` | Very shallow (4 layers), full width (768 dim) |
| 22 | a2 | 0.994955 | ok | `7632c47` | Shallower/wider with HEAD_DIM=96 (a1's improvement) |
| 23 | a2 | 0.994254 | ok | `15fbcf1` | Shallower/wider with reduced WARMDOWN_RATIO=0.3 |
| 24 | a0 | 0.993903 | ok | `13b6067` | DEPTH=6, ASPECT_RATIO=128, EMBEDDING_LR=0.5 (fine-tune) |
| 25 | a2 | 1.017291 | ok | `2c596f6` | Shallower/wider with HEAD_DIM=80 |
| 26 | a0 | 0.992399 | ok | `109f16f` | DEPTH=6, MATRIX_LR=0.05 (increase matrix lr) |
| 27 | a2 | 1.025872 | ok | `8fb7601` | HEAD_DIM=80 + WARMDOWN=0.3 (tuning smaller heads) |
| 28 | a0 | 0.997339 | ok | `48f047e` | DEPTH=7, ASPECT_RATIO=96 (768 dim, 7 layers) |
| 29 | a2 | 1.042720 | ok | `814089f` | HEAD_DIM=104 + WARMDOWN=0.3 (interpolating heads) |
| 30 | a0 | 0.993162 | ok | `337b2ea` | DEPTH=6, WEIGHT_DECAY=0.15 |
| 31 | a2 | 0.998852 | ok | `d81b318` | HEAD_DIM=96, WARMDOWN=0.3, EMBEDDING_LR=0.7 (final optimization) |
| 32 | a0 | 1.000415 | ok | `8c48e43` | DEPTH=6, HEAD_DIM=96 (smaller head dimension) |
| 33 | a2 | 0.999176 | ok | `88beac0` | Best config + MATRIX_LR=0.05 (combining improvements) |
| 34 | a0 | 0.991730 | ok | `a17e2d7` | DEPTH=6, UNEMBEDDING_LR=0.006 |
| 35 | a0 | 0.991387 | ok | `b00ca9e` | DEPTH=6, UNEMBEDDING_LR=0.008 |
| 36 | a0 | 0.990882 | ok | `dee7ac2` | DEPTH=6, UNEMBEDDING_LR=0.010 (final) |

**Best so far: val_bpb 0.990882 by a0 at commit `dee7ac2`.**

## Findings (append-only broadcast)

- `#11` **a0** [commit `3ee9f66`]: Shallower model (6 layers × 768 dim) beats baseline: val_bpb 0.991949 vs 0.996598. Uses 50.5 GB VRAM vs 68 GB for baseline. Suggests shallower/wider is more efficient for this 5-min training window.
- `#27` **a1** [commit `82ce95e`]: Reducing HEAD_DIM from 128 to 96 achieves val_bpb 0.994809, beating baseline (0.996598) by 0.18%. Small attention head dimension appears beneficial for this 5-minute training window.
- `#41` **a2** [commit `15fbcf1`]: Shallower/wider + HEAD_DIM=96 + reduced warmdown: val_bpb 0.994254 beats baseline (0.996598) and adoption attempt (0.996702). HEAD_DIM reduction from 128 to 96 provides 0.002 improvement. Warmdown schedule reduction (0.5→0.3) adds another 0.0007 improvement.
- `#47` **a2** [commit `15fbcf1`]: Final best: DEPTH=6, ASPECT_RATIO=128, HEAD_DIM=96, WARMDOWN=0.3 yields val_bpb 0.994254. Systematic exploration of 12 runs tested optimizer tuning and architectural variants. HEAD_DIM sweet spot at 96 (80 too small: 1.017, 104 too large: 1.042). Additional optimizer adjustments (EMBEDDING_LR=0.7, MATRIX_LR=0.05) did not improve final result, suggesting found config is locally optimal.
- `#52` **a0** [commit `dee7ac2`]: Best result: val_bpb 0.990882 with DEPTH=6, ASPECT_RATIO=128, UNEMBEDDING_LR=0.010. Sweet spot found: depth=6 is optimal (4,5 layers too shallow, 7+ layers hurt); UNEMBEDDING_LR range 0.006-0.010 works well; original 0.004 was too conservative.

## Disconfirmations (negative results, attempts to falsify)

- `#8` **a0** [commit `f690b02`]: Deeper/narrower (10×640) did not improve over baseline: val_bpb 1.000690 vs baseline 0.996598. Suggests current depth/width balance is already near-optimal or wider models are better with limited training time.
- `#30` **a0** [commit `bd02971`]: 5-layer model (1.039023) performs worse than baseline, following similar pattern to 4-layer and 10-layer. Suggests 6 layers is optimal depth for this architecture in 5-minute window.
- `#31` **a0** [commit `60031ee`]: 4-layer model (1.037945) is too shallow. Combined with 5-layer result, confirms 6-layer depth is near-optimal.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

