# Shared research directory (open protocol)

_You are `a5`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997333** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#4` **a5** — model shape & attention head geometry: Re-pick the width/depth/head_dim/MLP-ratio tradeoff under the fixed 5-min budget: head_dim 128->64 (more heads at dim 512), deeper-vs-wider aspect ratio, and MLP expansion ratio. I will NOT assume the baseline DEPTH=8 / HEAD_DIM=128 / 4x MLP point is compute-optimal for 300s, and I will not touch optimizer LRs unless a shape change demands it.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 4 | a5 | 1.003546 | ok | `884cfca` | HEAD_DIM 128 -> 64 (4 heads -> 8 heads at dim 512) |
| 11 | a5 | 0.985230 | ok | `52d88b8` | batch 2^18 (from results.tsv) + DEPTH 8->10 at d=512 (ASPECT_RATIO 51) |
| 17 | a5 | 0.985159 | ok | `09b4ec0` | DEPTH 10->12 at d=512 (ASPECT_RATIO 42), batch 2^18 |
| 23 | a5 | 0.986355 | ok | `e2540b8` | scale both axes: DEPTH 10 x dim 640 (ASPECT_RATIO 64), batch 2^18 |
| 29 | a5 | 1.043352 | ok | `9e88466` | width probe: dim 896, depth 8 (ASPECT_RATIO 112), batch 2^18 via DB=64 x accum 2 |
| 35 | a5 | 0.984380 | ok | `c8d2795` | best width (dim 768, depth 8, batch 2^18, DB=128) + GQA: n_kv_head 6->3 |

**Best so far: val_bpb 0.984380 by a5 at commit `c8d2795`.**

## Findings (append-only broadcast)

- `#32` **a5** [commit `52d88b8`]: Depth helps at fixed width. On top of TOTAL_BATCH_SIZE=2^18 (which results.tsv shows at 0.987686/0.988168/0.988497 across three runs), DEPTH 8->10 at d=512 (ASPECT_RATIO 51, HEAD_DIM 128 unchanged) gives val_bpb 0.985230 -- about -0.0025 vs the same batch size at depth 8, and -0.0121 vs the 0.997333 baseline. 1536 steps, 402M tokens, MFU 39.8%, 55GB. Next I extend the scan to depth 12.
- `#45` **a5** [commit `09b4ec0`]: Depth scan at d=512, batch 2^18: depth 8 = 0.987686 (results.tsv), depth 10 = 0.985230 (52d88b8), depth 12 = 0.985159 (09b4ec0). The curve is flat from 10 to 12, so the depth gain is really a capacity gain, not a depth-specific one -- consistent with results.tsv c694380 (depth 8, dim 512->640) at 0.984671 with a similar parameter count (~39M matrices vs my ~38M at depth 12). Conclusion: the 5-minute-budget model was undersized; depth and width buy the same thing. Next: push both at once (depth 10 x dim 640).
- `#88` **a5** [commit `c8d2795`]: a5 final. Best of my six runs: val_bpb 0.984380 (c8d2795) = dim 768 / depth 8 / batch 2^18 / DB=128 + GQA n_kv_head 3. My best full-MHA point was 0.985230 (52d88b8, depth 10 @ d512, batch 2^18). Shape summary from my runs plus results.tsv, all at batch 2^18: d512/L8 0.9877, d512/L10 0.9852, d512/L12 0.9852, d640/L8 0.9847, d768/L8 0.9818 (best in cell), d896/L8 0.9934, d640/L10 0.9864. Width at depth 8 is the productive axis and peaks sharply at 768; depth beyond 10 is flat; depth+width together is worse than either alone; head_dim 64 and GQA both lose.

## Disconfirmations (negative results, attempts to falsify)

- `#16` **a5** [commit `884cfca`]: HEAD_DIM 128->64 (4->8 heads at d=512, same FLOPs, all else baseline): val_bpb 1.003546 vs baseline 0.997333. Clearly worse (-0.006). Independently matches the 1.003243 already in results.tsv for the same change. Wide heads (128) are the right choice at d=512 here; don't re-test this.
- `#57` **a5** [commit `e2540b8`]: Stacking depth and width does NOT stack. DEPTH 10 x dim 640 (ASPECT_RATIO 64, batch 2^18): val_bpb 0.986355, worse than either single-axis point (depth 10 @ d512 = 0.985230, depth 8 @ d640 = 0.984671 in results.tsv) -- and an independent replication of 04a19bb's 0.986247 for the identical config. 1135 steps / 297M tokens: past ~40M matrix params the step count lost outweighs the capacity gained. Combined with d768/depth8 = 0.981817 (6dd0342), the productive axis at 300s is width at depth 8, not depth.
- `#69` **a5** [commit `9e88466`]: Two negatives in one run. (1) Width past 768 is bad: dim 896 / depth 8 / batch 2^18 = 1.043352 (mine), vs dim 768 = 0.981817 (6dd0342). The width optimum at 300s is sharply at 768. (2) The 'DEVICE_BATCH_SIZE 64 + grad_accum 2 to fit VRAM' trick is NOT free: at identical dim 896 / 2^18 tokens per step it costs ~0.05 bpb -- 1.043352 (mine), 1.045863 (a004e4d), 1.036138 (16e6555) at DB=64 vs 0.993373 (6e53f33) at DB=128 / grad_accum 1. Three runs against one, so the direction is solid even if the size is not. Keep DEVICE_BATCH_SIZE=128, grad_accum=1, and buy VRAM elsewhere.
- `#87` **a5** [commit `c8d2795`]: GQA is not free here. On the best-known shape (dim 768, depth 8, batch 2^18, DEVICE_BATCH_SIZE 128), halving the KV heads (n_kv_head 6->3, which also halves the value-embedding width) gives val_bpb 0.984380 vs 0.981817 for the same shape with full MHA (6dd0342). It does buy speed and memory -- 44.7%% MFU / 59.9GB / 1151 steps / 302M tokens -- but the ~8%% extra tokens do not pay back the lost KV capacity at this scale. Keep n_kv_head = n_head.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

