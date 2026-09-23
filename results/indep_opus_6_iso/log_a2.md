# Shared research directory (open protocol)

_You are `a2`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#63` **a2** — compute-optimal model scaling: Treat the 5-min budget as a fixed FLOP budget and re-derive the width/depth/token tradeoff: at dim 768 x 8L the run sees only ~3-4 tokens/param, far under Chinchilla, so I will test narrower/shallower models that buy many more optimizer steps. I will not assume the current dim-768 depth-8 shape, HEAD_DIM 128, or the LR-scaling exponent are near-optimal.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 25 | a2 | 1.010895 | ok | `701da24` | shrink width: dim 768 -> 512 (depth 8), buying ~1.7x more optimizer steps |
| 27 | a2 | 1.034067 | ok | `646fa8b` | overlap CPU dataloader with GPU compute via background prefetch thread on its own CUDA stream |
| 29 | a2 | 1.006761 | ok | `7bc0558` | cut dataloader packing buffer 1000 -> 192 to remove the per-step Python best-fit stall |
| 31 | a2 | 1.032114 | ok | `be8adb5` | halve tokens per optimizer step: 2^17 -> 2^16 (device batch 64 -> 32) to buy more Muon steps |
| 33 | a2 | 1.005200 | ok | `030af2e` | sync GPU once per 4 steps instead of every step, so the Python dataloader and launches overlap with compute |
| 35 | a2 | 1.007255 | ok | `b8f4560` | head_dim 128 -> 64 at dim 512 (4 heads -> 8 heads), same params and FLOPs |

**Best so far: val_bpb 1.005200 by a2 at commit `030af2e`.**

## Findings (append-only broadcast)

- `#74` **a2** [commit `7bc0558`]: Two measured gains over the dim-768/depth-8 base (1.012347): (1) dim 768 -> 512 at depth 8 gives val_bpb 1.010895 (2398 steps, 314M tokens); shape is nearly flat here because the step is not purely GPU-bound. (2) Passing buffer_size=192 instead of the default 1000 to make_dataloader (its best-fit packing scan is O(B*buffer_size) pure Python, run every step) gives val_bpb 1.006761 at 2577 steps / 337.8M tokens, MFU 25.2 -> 27.1. One line, no model change, no eval change.
- `#82` **a2** [commit `030af2e`]: Removing the per-step GPU sync helps a little: syncing once per 4 steps (block timing, loss/NaN check at block boundaries, ~10 lines in the training loop) gives 346.6M tokens / 2644 steps vs 337.8M / 2577, MFU 27.1 -> 27.8, val_bpb 1.006761 -> 1.005200. Smaller than I expected, which says the ~48ms/step fixed cost is GPU-side (launch-bound small kernels), not CPU-side: the dataloader was already overlapping with queued GPU work.
- `#87` **a2** [commit `030af2e`]: FINAL (a2): best val_bpb 1.005200 at commit 030af2e, from the 1.012347 start. Stack: dim 512 depth 8 (head_dim 128, 4 heads) + make_dataloader(buffer_size=192) + one GPU sync per 4 steps instead of two per step. 2644 steps, 346.6M tokens, 27.8% MFU, 22.7GB. Main lesson for the group: this 5-min setup is per-step-overhead-bound (MFU ~28%), so shape/head_dim changes are nearly flat and throughput changes are where the gradient is; the untested big item is the fp32 logits+softcap+CE chain over 131072x8192 and cudagraphs.

## Disconfirmations (negative results, attempts to falsify)

- `#69` **a2** [commit `646fa8b`]: Background-thread prefetch of the dataloader HURTS: 1594 steps / 16.7% MFU vs 2398 steps / 25.2% MFU for the identical model without it (val_bpb 1.034067 vs 1.010895). Per-step losses match exactly, so it is pure throughput loss: the dataloader is pure Python, so a worker thread just contends for the GIL with the main thread's kernel-launch path. Do not try to hide the loader behind compute with threads.
- `#78` **a2** [commit `be8adb5`]: Halving tokens/step (2^17 -> 2^16, device batch 32) is clearly worse: val_bpb 1.032114 vs 1.006761 for the identical model at 2^17. Steps only rose 2577 -> 3638 (+41%, not +100%) and throughput fell 337.8M -> 238.4M tokens (MFU 27.1 -> 19.2). Implied per-step fixed cost ~48ms out of a 116ms step at 2^17: this setup is dominated by per-step CPU/launch overhead, not by model FLOPs.
- `#86` **a2** [commit `b8f4560`]: head_dim 64 (8 heads) at dim 512 is slightly worse than head_dim 128 (4 heads): val_bpb 1.007255 vs 1.005200, despite being marginally faster (2680 steps / 351.3M tokens vs 2644 / 346.6M, MFU 28.2 vs 27.8). Same params, same FLOPs. So few wide heads are fine here; head_dim is not a lever worth another run.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

