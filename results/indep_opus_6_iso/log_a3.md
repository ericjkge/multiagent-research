# Shared research directory (open protocol)

_You are `a3`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#4` **a3** — compute allocation: batch size x model shape: Re-tune the token/step tradeoff: how many optimizer steps the 5 minutes buys (TOTAL_BATCH_SIZE) and how to reinvest saved compute into width/depth and cheaper value-embedding budget. I do not assume the baseline's 524K-token batch, depth-8/dim-512 shape, or 4 full-vocab value embeddings are optimal.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 3 | a3 | 1.000800 | ok | `440abfc` | halve TOTAL_BATCH_SIZE to 2**18 (2x optimizer steps, grad_accum=1) |
| 7 | a3 | 1.010210 | ok | `81ca2ea` | TOTAL_BATCH_SIZE 2**17, device batch 64 (4x steps vs baseline) |
| 11 | a3 | 0.988145 | ok | `6f458d9` | depth 8->10 (dim 512->640) at batch 2**18, device batch 64 (grad_accum 2) |
| 15 | a3 | 0.994653 | ok | `a0e20cf` | depth 10->12 (dim 640->768) at batch 2**18, device batch 64 |
| 19 | a3 | 0.991023 | ok | `f718c60` | depth 10 widened: ASPECT_RATIO 64->76 (dim 640->768) at batch 2**18 — width vs depth at equal dim |
| 23 | a3 | 0.990440 | ok | `6a5be8e` | pure depth test: depth 12 at fixed dim 640 (ASPECT_RATIO 53), batch 2**18 |

**Best so far: val_bpb 0.988145 by a3 at commit `6f458d9`.**

## Findings (append-only broadcast)

- `#11` **a3** [commit `440abfc`]: TOTAL_BATCH_SIZE 2**19 -> 2**18 (grad_accum 2 -> 1, same device batch 128): val_bpb 1.012347 -> 1.000800 (-0.0115). The 5-min budget is step-starved at 524K tokens/step; doubling optimizer steps at half the tokens/step is a clear win with no other change (no LR retune).
- `#27` **a3** [commit `6f458d9`]: Scaling the model up is the big lever once batch is right: depth 8->10 (dim 512->640, ASPECT_RATIO 64) at TOTAL_BATCH_SIZE 2**18 with device batch 64 gives val_bpb 0.988145, vs 1.000800 at depth 8 / same batch and 1.012347 baseline. Peak VRAM only 34.6GB, so there is room to go bigger.
- `#43` **a3** [commit `f718c60`]: Width is not the productive axis: at depth 10, widening dim 640->768 (ASPECT_RATIO 64->76) gives val_bpb 0.991023, worse than 0.988145 at dim 640. Combined with depth 12/dim 768 = 0.994653, the ordering at batch 2**18 is d10/640 < d10/768 < d12/768, i.e. the gain from 8->10 layers came from depth, not from the width that came with it.
- `#56` **a3** [commit `6f458d9`]: FINAL (a3): best val_bpb 0.988145 at commit 6f458d9 = baseline + TOTAL_BATCH_SIZE 2**18 (device batch 64, grad_accum 2) + DEPTH 10 (dim 640, ASPECT_RATIO 64 unchanged). That is -0.0242 vs the 1.012347 baseline, from two config lines only, no new code. Map of the lattice I measured at batch 2**18: d8/512=1.000800, d10/640=0.988145, d10/768=0.991023, d12/640=0.990440, d12/768=0.994653; batch 2**17 at d8=1.010210.

## Disconfirmations (negative results, attempts to falsify)

- `#18` **a3** [commit `81ca2ea`]: Going further down in batch size does NOT keep helping: TOTAL_BATCH_SIZE 2**17 (device batch 64, 4x baseline steps) gives val_bpb 1.010210, clearly worse than 2**18's 1.000800. Optimum is around 2**18 = 262K tokens/step. VRAM at 2**17/dev64 is only 22.7GB vs 44.9GB at 2**18/dev128, so there is headroom to spend on model size.
- `#35` **a3** [commit `a0e20cf`]: Scaling past depth 10 overshoots: depth 12 (dim 768) at batch 2**18 / device batch 64 gives val_bpb 0.994653, worse than depth 10 / dim 640 (0.988145) though still better than depth 8 (1.000800). The depth-vs-steps optimum at a 5-min budget and 262K-token batches is around depth 10.
- `#55` **a3** [commit `6a5be8e`]: Pure depth also peaks at 10: depth 12 at fixed dim 640 (ASPECT_RATIO 53) gives val_bpb 0.990440, worse than depth 10/dim 640 (0.988145) and only marginally better than depth 12/dim 768 (0.994653). So neither extra depth nor extra width beyond 10x640 pays off in 5 minutes at 262K-token batches; 10x640 is a genuine local optimum on this lattice.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

