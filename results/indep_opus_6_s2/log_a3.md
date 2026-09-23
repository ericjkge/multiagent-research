# Shared research directory (open protocol)

_You are `a3`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997333** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#7` **a3** — compute allocation: batch size x model shape: Jointly retune the tokens-per-step / number-of-steps tradeoff (TOTAL_BATCH_SIZE) with model depth+width and the LRs that must co-scale with them. I will not assume the shipped TOTAL_BATCH_SIZE=2^19 or DEPTH=8 are jointly optimal for a 300s budget, and I will not touch the attention/optimizer internals unless the shape experiments plateau.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 6 | a3 | 0.988497 | ok | `364f25b` | TOTAL_BATCH_SIZE 2**19 -> 2**18 (halve tokens/step, double optimizer steps) |
| 15 | a3 | 0.989854 | ok | `f93f367` | U-net style learnable skip connections (2nd half blocks read 1st half outputs), at batch 2**18 |
| 21 | a3 | 0.986247 | ok | `04a19bb` | stack both measured capacity wins: DEPTH 8->10 at ASPECT_RATIO 64 (dim 640, 5 heads, 10 layers), batch 2**18/DB128 |
| 27 | a3 | 0.993373 | ok | `6e53f33` | width ladder step 4: ASPECT_RATIO 112 (dim 896, 7 heads), depth 8, batch 2**18 at DEVICE_BATCH_SIZE=128 (grad_accum 1) |
| 33 | a3 | 0.983388 | ok | `926d1bf` | best shape (dim 768, depth 8, batch 2**18/DB128) + WINDOW_PATTERN all-L (every layer full 2048 ctx instead of SSSL) |
| 36 | a3 | 0.982139 | ok | `d68fad3` | best shape (dim 768, depth 8, SSSL, batch 2**18/DB128) + value embeddings on ALL layers instead of alternating (free capacity, ~no FLOPs) |

**Best so far: val_bpb 0.982139 by a3 at commit `d68fad3`.**

## Findings (append-only broadcast)

- `#19` **a3** [commit `364f25b`]: TOTAL_BATCH_SIZE 2**19 -> 2**18 (DEVICE_BATCH_SIZE stays 128, grad_accum 2->1): val_bpb 0.988497 vs baseline 0.997333, i.e. -0.0088. 1821 steps / 477M tokens, MFU 38.2%, 44.9GB. Optimizer-step count, not tokens seen, is the binding constraint at 300s.
- `#53` **a3** [commit `04a19bb`]: Depth does NOT pay at dim 640: DEPTH 8->10 at ASPECT_RATIO 64 (dim 640, 10 layers, 85.9M params, batch 2**18/DB128) gives val_bpb 0.986247, worse than the same width at depth 8 (c694380, 0.984671) and much worse than depth 8 at dim 768 (6dd0342, 0.981817). Only 1137 steps vs 1821. Combined with 09b4ec0 (depth 12 @ d512, 0.985159 ~= depth 10 @ d512, 0.985230), the width ladder 512->640->768 (0.9885 -> 0.9847 -> 0.9818) is the productive axis and extra depth is not. Next I test dim 896 at DEVICE_BATCH_SIZE=128 (grad_accum 1).
- `#65` **a3** [commit `6e53f33`]: Width ladder tops out at dim 768. ASPECT_RATIO 112 (dim 896, 7 heads, 121M params, depth 8, batch 2**18) at DEVICE_BATCH_SIZE=128/grad_accum=1: val_bpb 0.993373, 836 steps, 75.8GB, MFU 44.9%. Ladder at depth 8 / batch 2**18 / DB128 is now 512:0.9885, 640:0.9847, 768:0.9818 (6dd0342), 896:0.9934 -> clear optimum at dim 768. Also note a004e4d ran the same dim 896 at DEVICE_BATCH_SIZE=64 and got 1.0459, i.e. 0.05 worse than the identical width at DB=128, which is further evidence that grad_accum must stay 1.
- `#91` **a3** [commit `926d1bf`]: Attention window length is near-optimal as shipped. At the best shape (dim 768, depth 8, batch 2**18/DB128): WINDOW_PATTERN all-L (every layer full 2048 ctx) gives 0.983388 (1071 steps, MFU 49.5%) vs 0.981817 for SSSL (6dd0342). Combined with 4fa4d45 (short window seq/4 at d512: 0.996114 vs 0.988497), both lengthening and shortening the windows are worse, so SSSL sits at the optimum and the curve is asymmetric - shortening costs ~5x more than lengthening.
- `#92` **a3** [commit `d68fad3`]: FINAL / my best: value embeddings on ALL 8 layers instead of alternating, at dim 768 / depth 8 / SSSL / batch 2**18 / DB128 -> val_bpb 0.982139 (119.5M params, 1071 steps, 67.2GB). That is a tie with the same config using alternating VE (6dd0342, 0.981817) - inside the ~0.001 run-to-run spread - so the 4 extra embedding tables (+25M params, ~3% slower steps) buy nothing: VE capacity is already saturated at 4 of 8 layers. My 6 runs: 0.988497 (batch 2**18 at DB128), 0.989854 (U-net skips, worse), 0.986247 (dim 640 x depth 10), 0.993373 (dim 896 at DB128), 0.983388 (all-L windows), 0.982139 (VE on all layers). Best cell config remains 6dd0342 at 0.981817.

## Disconfirmations (negative results, attempts to falsify)

- `#29` **a3** [commit `364f25b`]: [weak, from shared results.tsv, not my own runs] Aborted a queued 2^17 attempt before it claimed the GPU. Every 22.2GB run (DEVICE_BATCH_SIZE=64) in results.tsv lands at 0.9973-0.9990 whether TOTAL_BATCH is 2^18 (579a7ee 0.997353) or 2^17 (ab83fc3 0.997297, ef0b9e5 0.999011), while 2^18 at DEVICE_BATCH_SIZE=128 lands at 0.9877-0.9885 (51261c1, f211b86, 364f25b). The win is therefore not 'fewer tokens per optimizer step' alone; halving the per-device batch gives the whole gain back. 2^18 x DB=128 (grad_accum=1) is the local optimum on this axis.
- `#41` **a3** [commit `f93f367`]: U-net style learnable skip connections (2nd-half blocks add skip_lambda * matching 1st-half block output, init 0.5, in the fast scalar Adam group) at batch 2**18/DB=128: val_bpb 0.989854 vs 0.988497 for the identical config without them (364f25b). Slightly WORSE, and outside the ~0.0008 spread seen across the three duplicate 2**18 runs. This codebase already has per-layer resid_lambdas + x0_lambdas feeding the embedding into every block, which appears to cover what the U-net skips would add. Not worth the 8 lines.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

