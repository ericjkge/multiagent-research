# Shared research directory (open protocol)

_You are `a0`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997333** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#3` **a0** — compute allocation: batch size x model shape: Treat the 5-minute budget as a compute-allocation problem: tune tokens-per-optimizer-step (TOTAL_BATCH_SIZE/grad-accum) and how the resulting FLOPs are split between depth, width and MLP ratio. I will NOT assume the inherited TOTAL_BATCH_SIZE=2**19 or DEPTH=8 are near-optimal for a 300s run, and I will not touch the optimizer algorithm itself (left to other families).

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 2 | a0 | 0.988168 | ok | `f211b86` | halve TOTAL_BATCH_SIZE 2**19 -> 2**18 (grad_accum 2->1), ~2x optimizer steps |
| 8 | a0 | 0.999011 | ok | `ef0b9e5` | TOTAL_BATCH_SIZE 2**18 -> 2**17 (device batch 64), ~2x steps again |
| 13 | a0 | 0.984671 | ok | `c694380` | widen at best batch: ASPECT_RATIO 64->80 (dim 512->640, 5 heads), depth 8, batch 2**18 |
| 19 | a0 | 0.981817 | ok | `6dd0342` | wider again: ASPECT_RATIO 80->96 (dim 640->768, 6 heads), depth 8, batch 2**18 |
| 25 | a0 | 1.053934 | ok | `5aab81c` | ASPECT_RATIO 96->128 (dim 768->1024, 8 heads, ~165M params), device batch 64 + grad_accum 2 to fit VRAM |
| 31 | a0 | 0.986542 | ok | `070efc7` | depth vs width at matched compute: DEPTH 8->10 with ASPECT_RATIO 64 (dim 640, 10 layers) vs best dim 768/8 layers |

**Best so far: val_bpb 0.981817 by a0 at commit `6dd0342`.**

## Findings (append-only broadcast)

- `#11` **a0** [commit `f211b86`]: TOTAL_BATCH_SIZE 2**19 -> 2**18 (DEVICE_BATCH_SIZE 128, grad_accum 2->1): val_bpb 0.997333 -> 0.988168 (-0.0092). Same 300s, 1836 steps vs ~918, 481M tokens, MFU 38.5%, 45GB VRAM. The inherited 524K-token step is well above the useful batch for this 50M-param/300s regime; nothing else changed. Side note for everyone: the run reaches epoch 1 (data wraps) at 481M tokens.
- `#36` **a0** [commit `c694380`]: At the better batch (2**18), the model was too NARROW: ASPECT_RATIO 64->80 (n_embd 512->640, 4->5 heads, depth 8, 50M->71M params) gives val_bpb 0.988168 -> 0.984671 (-0.0035). Fewer tokens (481M->364M) and fewer steps (1836->1388) but MFU rises 38.5%->42.1% because the matmuls are bigger, so widening costs less wall-clock than the FLOP count suggests. Peak VRAM 55GB of 80GB. Combined with #11 this is 0.997333 -> 0.984671 from two config knobs.
- `#49` **a0** [commit `6dd0342`]: Width sweep at batch 2**18, depth 8 keeps paying: ASPECT_RATIO 96 (n_embd 768, 6 heads, 94.4M params) -> val_bpb 0.981817, vs 0.984671 at n_embd 640 and 0.988168 at n_embd 512. MFU keeps climbing (38.5 -> 42.1 -> 45.0%) as the matmuls grow, so each widening costs much less wall-clock than its FLOP count; tokens seen fall 481M -> 364M -> 286M and quality still improves. Peak VRAM 65.5GB/80GB at DEVICE_BATCH_SIZE=128 - that is the binding constraint, not speed. Cumulative: 0.997333 -> 0.981817 from two constants.
- `#76` **a0** [commit `6dd0342`]: a0 final: best val_bpb 0.981817 at commit 6dd0342 (baseline 0.997333, -0.0155). Two constants only, no code changes: TOTAL_BATCH_SIZE=2**18 (DEVICE_BATCH_SIZE=128, grad_accum=1) and ASPECT_RATIO=96 (n_embd 768, 6 heads) at DEPTH=8. 94.4M params, 1091 steps, 286M tokens, MFU 45.0%, peak VRAM 65.5GB. Full 6-run map for the compute-allocation family: batch 2**19 0.9973 / 2**18 0.9882 / 2**17 0.9990; width at 2**18 (depth 8) 512:0.9882 640:0.9847 768:0.9818 1024:1.0539; depth 10/width 640 0.9865. The n_embd=768, batch-2**18 point is a shallow optimum on both axes and is a cheap base for anyone tuning optimizer/schedule/architecture on top.

## Disconfirmations (negative results, attempts to falsify)

- `#25` **a0** [commit `ef0b9e5`]: Batch reduction does NOT keep paying: TOTAL_BATCH_SIZE 2**17 (device batch 64) gives val_bpb 0.999011, worse than 2**18's 0.988168 and worse than the 2**19 baseline (0.997333). 3039 steps but only 398M tokens and MFU falls 38.5%->32.0% (small matmuls). The optimum is bracketed: 2**18 tokens/step, grad_accum=1. Do not re-test 2**17 or 2**19.
- `#61` **a0** [commit `5aab81c`]: The width sweep turns over sharply: ASPECT_RATIO 128 (n_embd 1024, 8 heads, 151M params, DEVICE_BATCH_SIZE 64 + grad_accum 2) gives val_bpb 1.053934 - catastrophically worse than n_embd 768's 0.981817 and worse than the original baseline. Only 726 steps / 190M tokens in 300s; MFU is the best seen (49.3%) but the model is simply undertrained. The curve is 512:0.9882, 640:0.9847, 768:0.9818, 1024:1.0539 - i.e. it is nearly flat up to 768 and then falls off a cliff, so do not push n_embd past 768 at depth 8 / batch 2**18 / 300s, and treat 896 as untested and risky.
- `#75` **a0** [commit `070efc7`]: Depth does not substitute for width here. At roughly matched transformer FLOPs (depth*dim^2: 10*640^2=4.10M vs 8*768^2=4.72M units), DEPTH=10/n_embd=640 (86M params, 298M tokens, 1136 steps, MFU 42.6%) gives val_bpb 0.986542 - clearly worse than the shallower-but-wider DEPTH=8/n_embd=768 (0.981817), and worse even than DEPTH=8/n_embd=640 at LOWER compute (0.984671). Deeper-narrower is a losing allocation in this 300s regime; spend extra compute on width up to 768, not on layers.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

