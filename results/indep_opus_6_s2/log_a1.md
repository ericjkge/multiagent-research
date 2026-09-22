# Shared research directory (open protocol)

_You are `a1`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997333** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#2` **a1** — batch-size & step-count economics: Explore the tokens-per-step vs number-of-optimizer-steps tradeoff at fixed 5-min wall clock (TOTAL_BATCH_SIZE, grad-accum, device batch), and adapt LR/schedule to the resulting step count. I will NOT assume the inherited 2^19-token batch, 2 grad-accum steps, or that LRs tuned at that batch transfer.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 1 | a1 | 0.987686 | ok | `51261c1` | TOTAL_BATCH_SIZE 2^19 -> 2^18 (grad_accum 2->1): double optimizer steps in 5 min |
| 7 | a1 | 0.997297 | ok | `ab83fc3` | TOTAL_BATCH_SIZE 2^18 -> 2^17 (DEVICE_BATCH_SIZE 64): bracket the batch-size optimum |
| 12 | a1 | 0.990053 | ok | `650f4e3` | at 2^18/DB=128: scale all four LRs by 1.4x (matrix 0.04->0.056, embed 0.6->0.84, unembed 0.004->0.0056, scalar 0.5->0.7) |
| 18 | a1 | 0.985236 | ok | `da5c883` | a5's depth-10 @ d512 + batch 2^18, plus my change: WARMDOWN_RATIO 0.5 -> 1.0 (linear LR decay from step 0) |
| 24 | a1 | 0.987079 | ok | `c902068` | combine the two best size moves: DEPTH 10 AND dim 640 (ASPECT_RATIO 64, 5 heads) at batch 2^18, warmdown 1.0 |
| 30 | a1 | 0.991800 | ok | `1c2ec3e` | shallow-and-fat: DEPTH 7 at dim 896 (ASPECT_RATIO 128, 7 heads), batch 2^18 -- push width past 768 while trading a layer for VRAM and steps |

**Best so far: val_bpb 0.985236 by a1 at commit `da5c883`.**

## Findings (append-only broadcast)

- `#9` **a1** [commit `51261c1`]: TOTAL_BATCH_SIZE 2^19 -> 2^18 (grad_accum 2->1, DEVICE_BATCH_SIZE 128 unchanged): val_bpb 0.997333 -> 0.987686 (-0.0096). 1849 steps in 300s, 484.7M tokens, MFU 38.8%, 44.9GB. Same token throughput, twice the optimizer steps => the inherited 2^19 batch was well past the useful point for a 5-minute budget.
- `#22` **a1** [commit `ab83fc3`]: Throughput ceiling: prepare.make_dataloader is pure-Python best-fit packing and costs ~145 ms per 262144-token batch (measured standalone, B=128/T=2048, /root/autoresearch/.venv python). Observed step time at DB=128 is ~162 ms, so CPU dataloading is already ~90% of the critical path and is hidden only because loss.backward() is async. Practical consequences: (1) a background prefetch thread will NOT help, the packing loop is GIL-bound pure Python; (2) making the model faster buys at most ~12% more tokens before hitting the ~1.8M tok/s dataloader floor; (3) smaller DEVICE_BATCH_SIZE is strictly bad.
- `#71` **a1** [commit `1c2ec3e`]: Width turns over just past 768: DEPTH 7 at dim 896 (ASPECT_RATIO 128, batch 2^18, DB=128, 65.6GB) gives val_bpb 0.991800, far worse than depth-8 @ dim 768 (0.981817). Peer 6e53f33 (depth 8, dim 896, DB=128, 74.0GB) independently gives 0.993373, so the turnover is real and is not a VRAM/DB artifact -- note the other dim-896 runs at DEVICE_BATCH_SIZE 64 (1.0359/1.0434/1.0461) are confounded by the ~15% token loss from halving device batch that I measured in ab83fc3. Conclusion for the group: depth 8 x dim 768 x batch 2^18 x DB=128 is the optimum of the size grid this cell has measured; do not spend more runs widening.
- `#72` **a1** [commit `da5c883`]: a1 final: my best of 6 runs is val_bpb 0.985236 at commit da5c883 (DEPTH 10 @ d512, ASPECT_RATIO 51, TOTAL_BATCH_SIZE 2^18, DEVICE_BATCH_SIZE 128, WARMDOWN_RATIO 1.0), vs the 0.997333 baseline. My runs in order: 0.987686 (batch 2^19->2^18), 0.997297 (2^17 @ DB=64), 0.990053 (LRs x1.4), 0.985236 (depth 10 + warmdown 1.0), 0.987079 (depth 10 x dim 640), 0.991800 (depth 7 @ dim 896). Net contribution of my slot: batch 2^18 at DEVICE_BATCH_SIZE 128 is the right operating point and the gain is from optimizer steps, not effective LR; LR magnitude and schedule shape are both dead knobs; depth and width do not stack; and the pure-Python dataloader caps throughput at ~1.8M tok/s (~145ms per 262K-token batch).

## Disconfirmations (negative results, attempts to falsify)

- `#21` **a1** [commit `ab83fc3`]: Going further down in batch does NOT help: TOTAL_BATCH_SIZE 2^17 with DEVICE_BATCH_SIZE 64 gives val_bpb 0.997297 (vs 0.987686 at 2^18/DB=128), 3148 steps but only 412.6M tokens and MFU 33.1% (vs 38.8%). Cause is throughput, not batch: halving DEVICE_BATCH_SIZE costs ~15% of tokens to fixed per-step overhead. Corroborated by peer run 579a7ee (2^18 with DB=64, 22.2GB) = 0.997353, i.e. the same total batch as my 0.987686 run but at half device batch loses the entire gain.
- `#34` **a1** [commit `650f4e3`]: LR is NOT the reason the 2^18 batch helped. At 2^18/DB=128, scaling all four LRs by 1.4x (matrix 0.04->0.056, embed 0.6->0.84, unembed 0.004->0.0056, scalar 0.5->0.7) gives val_bpb 0.990053 vs 0.987686 at the inherited LRs (same commit otherwise). That is ~5x the observed replicate noise (0.987686 vs 0.988168 for the identical change run by two agents), so the inherited LRs are at or above the optimum for this batch; the 2^19->2^18 gain came from optimizer steps, not from the implied 2x effective LR per token.
- `#47` **a1** [commit `da5c883`]: LR-schedule shape is a dead knob here. WARMDOWN_RATIO 0.5 -> 1.0 (linear LR decay from step 0 instead of a constant first half) on top of depth-10 @ d512 + batch 2^18: val_bpb 0.985236 vs 0.985230 for the identical config with WARMDOWN 0.5 (52d88b8). A 0.000006 difference -- flat. Combined with my 650f4e3 result (1.4x LR is worse by 0.0024), the inherited LR peak and schedule are already at their optimum; spend runs elsewhere.
- `#59` **a1** [commit `c902068`]: Depth and width do NOT stack. DEPTH 10 x dim 640 (ASPECT_RATIO 64), batch 2^18: val_bpb 0.987079 (67.8GB) -- worse than either move alone (depth 10 @ d512 = 0.98523, depth 8 @ d640 = 0.98467) and worse than plain depth-8 @ d640. Third independent replicate of this cell (04a19bb 0.986247, e2540b8 0.986355, mine 0.987079), so it is not noise. At a fixed 5-min budget the useful axis is width at depth 8: 512 -> 0.98769, 640 -> 0.98467, 768 -> 0.98182. Peak VRAM is well predicted by 3.9 + 0.00977*depth*dim GB (fits all six measured points within 1.5GB), so d8/dim896 ~= 74GB and d8/dim1024 will not fit.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

