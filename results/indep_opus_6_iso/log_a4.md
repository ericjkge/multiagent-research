# Shared research directory (open protocol)

_You are `a4`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#5` **a4** — compute allocation: batch-size/step-count economics and model shape (depth, width, MLP ratio): I will tune where the fixed 5-minute FLOP budget goes: number of optimizer steps (global batch size) vs model size/shape (depth, aspect ratio, MLP expansion, head_dim). I will NOT assume the baseline's depth-8/dim-512/4x-MLP/524k-token-batch is optimal, and I will not modify optimizer internals or the attention kernel.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 4 | a4 | 1.000781 | ok | `0f98043` | halve global batch to 2^18 tokens (2x optimizer steps), LRs unchanged |
| 8 | a4 | 1.006829 | ok | `51ac1a0` | batch 2^17 tokens (device batch 64): push step count further |
| 12 | a4 | 1.010004 | ok | `21bd06a` | head_dim 64 (8 heads instead of 4) at dim 512, batch 2^18 |
| 16 | a4 | 0.986041 | ok | `df2eaa7` | widen model to dim 640 (aspect ratio 80, 5 heads x 128) at depth 8, batch 2^18 |
| 20 | a4 | 0.982780 | ok | `6dd7941` | widen further to dim 768 (aspect ratio 96, 6 heads x 128), batch 2^18 via device batch 64 x2 |
| 24 | a4 | 0.982146 | ok | `d99165f` | at dim 768: halve batch to 2^17 to recover optimizer steps (1072 -> ~2100) |

**Best so far: val_bpb 0.982146 by a4 at commit `d99165f`.**

## Findings (append-only broadcast)

- `#13` **a4** [commit `0f98043`]: Halving the global batch from 2^19 to 2^18 tokens (device batch 128, grad_accum 2->1, all LRs unchanged) gives val_bpb 1.000781 vs baseline 1.012347 (-0.0116). In a 300s budget the baseline is step-starved; doubling optimizer steps beats the larger batch.
- `#37` **a4** [commit `df2eaa7`]: Widening to dim 640 (ASPECT_RATIO 80, 5 heads x 128) at depth 8 with batch 2^18 gives val_bpb 0.986041, best so far (-0.0147 vs the same config at dim 512, -0.0263 vs baseline 1.012347). 70.8M params, MFU 41.3%, 357M tokens, 55.2GB peak. Fewer tokens at higher MFU and more capacity is clearly the better trade here.
- `#45` **a4** [commit `6dd7941`]: dim 768 (ASPECT_RATIO 96, 6 heads x 128), depth 8, batch 2^18 (device batch 64 x2): val_bpb 0.982780, new best (-0.0033 vs dim 640, -0.0296 vs baseline). 94.4M params, MFU 44.2%, 281M tokens, 1072 steps, only 33.5GB peak VRAM (memory is dominated by device batch, not model width). Width gains are flattening: 512->640 gave -0.0147, 640->768 only -0.0033.
- `#59` **a4** _(weak claim)_ [commit `d99165f`]: Batch-size optimum interacts with model size, but weakly at dim 768: halving to 2^17 (2096 steps, 275M tokens) gives val_bpb 0.982146 vs 0.982780 at 2^18 (1072 steps) - a 0.0006 difference, within single-run noise. Contrast with dim 512, where 2^17 was 0.006 WORSE than 2^18. Practical read: anywhere in 2^17-2^18 is fine at dim 768; the big lever was width, not batch.
- `#60` **a4** [commit `d99165f`]: a4 final: best val_bpb 0.982146 (commit d99165f) = baseline 1.012347 minus 0.0302. Config: depth 8, dim 768 (ASPECT_RATIO 96, 6 heads x 128), batch 2^17 tokens, device batch 64, everything else untouched - three one-line hyperparameter edits, no new code. Ladder: 2^19->2^18 batch -0.0116; dim 512->640 -0.0147; 640->768 -0.0033; 2^18->2^17 -0.0006. Negative results: batch 2^17 at dim 512 (+0.006), head_dim 64 (+0.009, throughput loss).

## Disconfirmations (negative results, attempts to falsify)

- `#20` **a4** [commit `51ac1a0`]: Going further down in batch size hurts: 2^17 tokens/step (device batch 64, same LRs) gives val_bpb 1.006829, worse than 2^18's 1.000781 (still better than the 2^19 baseline 1.012347). The batch-size optimum in this 300s setup is around 2^18 tokens.
- `#29` **a4** [commit `21bd06a`]: head_dim 64 (8 heads at dim 512) instead of head_dim 128 is worse: val_bpb 1.010004 vs 1.000781 at the same batch 2^18. The damage is throughput, not per-token quality: MFU fell 40%->28.3% and the run covered only 354M tokens (1352 steps). Keep head_dim 128.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

