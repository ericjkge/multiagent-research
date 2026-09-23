# Shared research directory (open protocol)

_You are `a5`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#64` **a5** — compute allocation: model shape vs optimizer steps: I will treat the 8-layer / 768-dim / 4x-MLP shape as an unjustified prior and sweep where the FLOPs go (width, depth, MLP ratio, GQA, window pattern) so that more optimizer steps fit in the fixed 300s. I will not assume the current shape is near compute-optimal, and I will keep TOTAL_BATCH_SIZE=2**17 fixed as a control rather than re-tuning batch size.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 26 | a5 | 0.990125 | ok | `76033a9` | shrink width 768 -> 640 (ASPECT_RATIO 96->80), depth 8 unchanged: buy ~30% more optimizer steps |
| 28 | a5 | 1.007564 | ok | `8b6f6c0` | shrink width 640 -> 512 (ASPECT_RATIO 80->64), depth 8: continue the size sweep toward Chinchilla |
| 30 | a5 | 0.990844 | ok | `ece5fb0` | value embeddings on every layer (4 -> 8) at dim 640: near-FLOP-free capacity in the CPU-bound regime |
| 32 | a5 | 1.010386 | ok | `3a960bb` | halve batch 2**17 -> 2**16 (device batch 32) at 8x640: tokens are CPU-capped, so optimizer steps are nearly free |
| 34 | a5 | 0.997068 | ok | `c867ac3` | head_dim 128 -> 64 at dim 640 (5 heads -> 10 heads), FLOP-identical, everything else back to best config |
| 36 | a5 | 0.988531 | ok | `40cae8a` | depth 8 -> 10 at the winning width 640 (ASPECT_RATIO 64): add capacity via depth rather than width |

**Best so far: val_bpb 0.988531 by a5 at commit `40cae8a`.**

## Findings (append-only broadcast)

- `#67` **a5** [commit `76033a9`]: Width 768->640 at depth 8 (ASPECT_RATIO 96->80, batch 2**17 fixed): val_bpb 0.990125 vs cell baseline 1.012347. Steps 2409, 315.8M tokens, 36.7% MFU, 70.8M params (39.3M transformer matrices). The 8x768 shape was well past compute-optimal for a 300s budget: at 3.46e8 FLOPs/token the run sees ~7 tokens per flop-carrying param, vs Chinchilla ~20, so the model is still too big. Shrinking further next.
- `#72` **a5** [commit `8b6f6c0`]: Profiled the data pipeline (CPU only, no training run): make_dataloader costs ~113 ms median per 131072-token batch on this box -- tokenize+parquet ~46 ms, best-fit scan ~15-40 ms, row fill ~9 ms. Measured step times are 110-124 ms, so the 300 s budget is bound by the CPU dataloader, NOT the GPU (MFU 27-42%). Passing buffer_size=32..128 instead of the default 1000 only recovers ~14 ms, so packing quality is not worth trading away. Practical consequence: tokens/s is capped near 1.1-1.3M whatever the model, which is why 512 (1.115M tok/s) barely out-stepped 640 (1.052M tok/s) and lost on quality. The right move is to spend the idle GPU headroom on capacity, not to shrink.
- `#90` **a5** [commit `40cae8a`]: BEST (a5): depth 10 x dim 640, head_dim 128, batch 2**17 -> val_bpb 0.988531 (2142 steps, 280.8M tokens, 40.3% MFU, 85.9M params). Beats 8x640 (0.990125) even though it trades 11% of the tokens for 25% more transformer matrices, so capacity added via DEPTH pays off where capacity added via width and via extra value embeddings did not. Full a5 sweep at batch 2**17: 8x512 1.007564, 8x640 0.990125, 10x640 0.988531; head_dim 64 at 8x640 0.997068; VE-on-all-layers at 8x640 0.990844; batch 2**16 at 8x640 1.010386. Reproduce with commit 40cae8a.

## Disconfirmations (negative results, attempts to falsify)

- `#71` **a5** [commit `8b6f6c0`]: Shrinking further, width 640->512 at depth 8, is WORSE: val_bpb 1.007564 vs 0.990125 at 640. So the naive Chinchilla-style 'still 3x too big' argument does not hold here (the 21-25M value-embedding params carry capacity the FLOPs estimate ignores). Width optimum at depth 8 is ~640; stop shrinking width.
- `#76` **a5** [commit `ece5fb0`]: Value embeddings on every layer (8 instead of the alternating 4) at 8x640: val_bpb 0.990844 vs 0.990125 with the alternating pattern -- a wash, slightly worse, and it cost 2.7% of the steps (2343 vs 2409) and +21M params. 'Free' capacity in the CPU-bound regime does not buy quality here, which says the 8x640 model is already near its capacity optimum for a 300s budget. Reverting to alternating VE.
- `#80` **a5** [commit `3a960bb`]: Halving the batch again, 2**17 -> 2**16 (device batch 32) at 8x640: val_bpb 1.010386, much worse than 0.990125 at 2**17. Tokens fell 315.8M -> 252.4M while steps only rose 2409 -> 3851, i.e. the dataloader/step overhead is NOT purely per-token -- there is a large fixed per-step cost, so small batches burn throughput for sub-linear step gains. 2**17 looks like the batch optimum for this 300s budget; do not go smaller.
- `#84` **a5** [commit `c867ac3`]: head_dim 64 (10 heads) vs head_dim 128 (5 heads) at dim 640, FLOP-identical and throughput-identical (313.3M vs 315.8M tokens, 2390 vs 2409 steps): val_bpb 0.997068 vs 0.990125. Fewer, wider heads win here by 0.007 bpb. Splitting the same 640 channels into more heads is a clear loss, so HEAD_DIM=128 should stay.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

