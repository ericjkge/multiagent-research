# Shared research directory (open protocol)

_You are `a2`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997333** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#3` **a2** — compute allocation & throughput: Hold architecture and optimizer math fixed; buy more optimizer steps and more tokens inside the 300s: total/device batch size, grad-accum count, cheaper loss head (chunked/bf16 logits), MFU. I will NOT assume the baseline's 2^19-token batch or DEVICE_BATCH_SIZE=128 are near-optimal for a 5-minute budget.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 2 | a2 | 0.988312 | ok | `10ebdbc` | TOTAL_BATCH_SIZE 2^19 -> 2^18 (grad_accum 2 -> 1): 2x optimizer steps in 300s |
| 8 | a2 | 0.995947 | ok | `2b0e9e0` | TOTAL_BATCH_SIZE 2^17, DEVICE_BATCH 64 (grad_accum=1): 2x more steps again |
| 13 | a2 | 0.994703 | ok | `eb54697` | background-thread prefetch of the dataloader (overlap CPU packing with GPU), TOTAL_BATCH 2^18 |
| 18 | a2 | 0.988608 | ok | `2e7c056` | on a4's reshape base (c1f788e): WARMDOWN_RATIO 0.5 -> 0.35 (longer at peak LR) |
| 22 | a2 | 0.984440 | ok | `e85293e` | width-640 base (75a8d84) + WARMDOWN_RATIO 0.5 -> 0.65 (longer LR decay) |
| 28 | a2 | — | crash | `1475c37` | width 768 -> 896 (ASPECT_RATIO 112) at batch 2^18, with a chunked fp32 loss head to buy the VRAM |

**Best so far: val_bpb 0.984440 by a2 at commit `e85293e`.**

## Findings (append-only broadcast)

- `#11` **a2** [commit `10ebdbc`]: Halving the optimizer batch (TOTAL_BATCH_SIZE 2^19->2^18, grad_accum 2->1, nothing else changed) gives val_bpb 0.988312 vs baseline 0.997333 (-0.0090). 1822 steps, 477.6M tokens, mfu 38.25%, 44.9GB peak. Note the run reaches epoch 1, i.e. the 300s budget already wraps the train shard at this throughput.
- `#57` **a2** _(weak claim)_ [commit `e85293e`]: LR-schedule time allocation is flat near the top but falls off sharply on the short side: on the width-640 @2^18 base (75a8d84, 0.984898), WARMDOWN_RATIO 0.65 gives 0.984440 (-0.0005, i.e. within the ~0.0005 replication spread) while 0.35 gave +0.0019 on the reshape base. Conclusion: leave WARMDOWN_RATIO at 0.5-0.65 and do not spend runs there; shortening the decay is a real loss.
- `#72` **a2** [commit `e85293e`]: a2 final: best val_bpb 0.984440 at commit e85293e = width 640 (ASPECT_RATIO 80, depth 8) + TOTAL_BATCH_SIZE 2^18 (grad_accum 1) + WARMDOWN_RATIO 0.65; 362.5M tokens, 1383 steps, mfu 41.9%, 55.2GB. Summary of my six runs: batch 2^19->2^18 is worth -0.0090 and is the single biggest cheap win; 2^17 is worse (+0.008); background-thread dataloader prefetch is worse (+0.006, GIL contention, the loader is not the bottleneck); WARMDOWN 0.35 is worse (+0.0019) while 0.65 is neutral-to-slightly-better (-0.0005); width 896 OOMs at DEVICE_BATCH 128.

## Disconfirmations (negative results, attempts to falsify)

- `#25` **a2** [commit `2b0e9e0`]: Shrinking the batch further does NOT keep paying: TOTAL_BATCH_SIZE 2^17 + DEVICE_BATCH 64 gives val_bpb 0.995947, clearly worse than 2^18/128 (0.988312, same file otherwise). Throughput also fell (424.7M vs 477.6M tokens, mfu 34.1% vs 38.3%) and per-step dt was jittery (85-153ms for a 131K-token step that should cost ~80ms), i.e. the run is partly CPU/overhead-bound, not GPU-bound. 2^18 with grad_accum=1 looks like the optimum of this axis.
- `#36` **a2** [commit `eb54697`]: Dataloader prefetching on a background thread HURTS: 0.994703 vs 0.988312 for the same 2^18 config, and throughput fell (422.8M vs 477.6M tokens, mfu 33.8% vs 38.25%). A second Python thread doing the best-fit packing contends for the GIL with the main thread's kernel launches (and adds a per-batch clone + stream sync), so the CPU-side loader is not the thing limiting this run - do not spend a run on double-buffering the loader.
- `#47` **a2** [commit `2e7c056`]: Shortening the LR warmdown hurts: on the depth-10/MLP-3x/short-window-4 @2^18 base (c1f788e, 0.986692), WARMDOWN_RATIO 0.5 -> 0.35 gives 0.988608 (+0.0019), throughput unchanged (445M tokens, mfu 36.1%). More time at peak LR is not what this 300s budget wants; if anything the decay phase should be longer than half the run.
- `#71` **a2** [commit `1475c37`]: Width 896 (ASPECT_RATIO 112, depth 8, batch 2^18, DEVICE_BATCH 128) OOMs on the 80GB card: crash in backward at 79.15GB in use. I tried to buy the headroom with a chunked loss head (16 chunks of 16384 tokens, fp32 softcap+CE per chunk, verified numerically identical to the original mean/none reductions) and it was not enough - inductor keeps every chunk's logits alive for backward, so chunking removes the fp32 (B*T,V) copy but not the bf16 one. Anyone wanting to test width >=896 needs either DEVICE_BATCH 64 (which cost ~0.010 bpb at width 768 in dd3c58c, via lost throughput) or a genuinely recomputed/fused CE. The 512->640->768 width ladder (0.9883 -> 0.9849 -> 0.9820 at 2^18) is therefore unresolved above 768.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

