# Shared research directory (open protocol)

Baseline at start of the cell: **val_bpb 0.997359** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 36 runs. Used so far: a0: 0 left.

## Approaches (slots)

- `#2` **a0** — compute-allocation co-design: Treat the 5-min budget as a FLOP/bandwidth allocation problem: measure where the step time actually goes (output head vs body vs attention), then re-balance model size, batch size, head/MLP shapes and the loss computation to buy more effective tokens. I will not assume the shipped DEPTH=8 / dim=512 / 2^19-token batch is optimal, nor that the f32 softcapped logit head is free.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 1 | a0 | 1.002094 | ok | `a2b0086` | DEPTH=10 (dim 640, 5 heads) instead of DEPTH=8 |
| 2 | a0 | 1.004693 | ok | `ce452c5` | HEAD_DIM 128->64 (8 heads of 64 at dim 512, same FLOPs) |
| 3 | a0 | 0.989498 | ok | `fa0ae36` | TOTAL_BATCH_SIZE 2^19 -> 2^18 (grad_accum 2->1, ~2x optimizer steps) |
| 4 | a0 | 0.998166 | ok | `88efe7e` | TOTAL_BATCH_SIZE 2^17, DEVICE_BATCH_SIZE 64 (grad_accum 1) |
| 5 | a0 | 0.993275 | ok | `c54a1a1` | MATRIX_LR 0.04->0.055 at batch 2^18 |
| 6 | a0 | 0.996679 | ok | `f4c9912` | U-net style LIFO skip connections between layer halves (learned skip_lambdas, init 0.1) |
| 7 | a0 | 0.987884 | ok | `5ed5ca9` | short sliding window 1024 -> 512 (SSSL pattern), ~8% fewer attn FLOPs |
| 8 | a0 | 0.988849 | ok | `46ebd9a` | short sliding window 512 -> 256 |
| 9 | a0 | 0.989226 | ok | `7beb9eb` | WINDOW_PATTERN all-S (only final layer full-context), short=512 |
| 10 | a0 | 0.987388 | ok | `c69abf9` | WINDOW_PATTERN SSL (3 full-context layers of 8), short=512 |
| 11 | a0 | 0.988534 | ok | `fa9f875` | WINDOW_PATTERN SL (4 full-context layers of 8), short=512 |
| 12 | a0 | 0.986397 | ok | `bced6a4` | WARMDOWN_RATIO 0.5 -> 0.7 |
| 13 | a0 | 0.987757 | ok | `01d447d` | WARMDOWN_RATIO 0.7 -> 1.0 (pure linear decay) |
| 14 | a0 | 0.984833 | ok | `67cd043` | Value embeddings on ALL layers (was every other layer) |
| 15 | a0 | 0.988392 | ok | `a677bf9` | token smear at embedding: x += smear_lambda * prev_token_emb (learned scalar, init 0.3) |
| 16 | a0 | 0.985109 | ok | `1a68458` | UNEMBEDDING_LR 0.004 -> 0.008 |
| 17 | a0 | 0.985614 | ok | `209a3d8` | inductor coordinate_descent_tuning=True (tune bandwidth-bound pointwise kernels) |
| 18 | a0 | 0.990728 | ok | `1e6b010` | MLP expansion 4x -> 3x |
| 19 | a0 | 0.984049 | ok | `418d5cb` | MLP expansion 4x -> 5x |
| 20 | a0 | 0.983147 | ok | `6fc32ab` | MLP expansion 5x -> 6x |
| 21 | a0 | 0.983318 | ok | `4a8e88c` | MLP expansion 6x -> 8x |
| 22 | a0 | 0.989912 | ok | `6f973e4` | gated relu^2 MLP (hidden 4x, up+gate) - same params/FLOPs as ungated 6x |
| 23 | a0 | 0.983005 | ok | `15fcdd9` | DEPTH 8 -> 10 at fixed dim 512 (ASPECT_RATIO 51), MLP 6x |
| 24 | a0 | 0.982289 | ok | `156f18a` | MATRIX_LR 0.04 -> 0.03 at MLP 6x |
| 25 | a0 | 0.983141 | ok | `33d6cfd` | MATRIX_LR 0.03 -> 0.022 |
| 26 | a0 | 0.984308 | ok | `488dbec` | WEIGHT_DECAY 0.2 -> 0.4 |
| 27 | a0 | 0.983402 | ok | `4e8d329` | WEIGHT_DECAY 0.2 -> 0.1 |
| 28 | a0 | 0.982867 | ok | `b8017b1` | MLP 8x re-probe at MATRIX_LR 0.03 |
| 29 | a0 | 0.983091 | ok | `0914981` | EMBEDDING_LR 0.6 -> 0.9 |
| 30 | a0 | 0.985770 | ok | `49d825d` | logit softcap 15 -> 30 |
| 31 | a0 | 0.982384 | ok | `52e3cc9` | logit softcap 15 -> 10 |
| 32 | a0 | 0.981856 | ok | `a2d0782` | attention softmax_scale 0.088 (1\/sqrt(128)) -> 0.12 with QK-norm |
| 33 | a0 | 0.985007 | ok | `344073b` | attention softmax_scale 0.12 -> 0.17 |
| 34 | a0 | 0.982201 | ok | `cc6d42a` | WARMDOWN_RATIO 0.7 -> 0.85 at the new best config |
| 35 | a0 | 0.981181 | ok | `cede5ff` | EMBEDDING_LR 0.6 -> 0.45 |
| 36 | a0 | 0.980742 | ok | `0ad512b` | EMBEDDING_LR 0.45 -> 0.32 |

**Best so far: val_bpb 0.980742 by a0 at commit `0ad512b`.**

## Findings (append-only broadcast)

- `#8` **a0** [commit `fa0ae36`]: TOTAL_BATCH_SIZE 2^19 -> 2^18 (grad_accum 2->1, DEVICE_BATCH_SIZE still 128): val_bpb 0.989498 vs baseline 0.997359. Clear -0.0079. The shipped 524K-token batch is too large for a 25M-param body in a 5-min budget; halving it doubles optimizer steps at ~the same tokens/sec.
- `#16` **a0** [commit `5ed5ca9`]: Shrinking the SHORT sliding window from 1024 to 512 (pattern SSSL, long=2048) improves val_bpb 0.989498 -> 0.987884 on top of the 2^18 batch. Cheaper attention buys more steps and costs nothing in quality. Running best: 0.987884.
- `#22` **a0** [commit `c69abf9`]: Window-pattern sweep at short=512, depth 8: all-S (1 long layer) 0.989226, SSSL (2 long) 0.987884, SSL (3 long) 0.987388 <-- best, SL (4 long) 0.988534. Interior optimum at 3 of 8 layers full-context. Best so far: 0.987388 (batch 2^18 + short window 512 + SSL).
- `#25` **a0** [commit `bced6a4`]: WARMDOWN_RATIO sweep (no warmup, final_lr 0): 0.5 -> 0.987388, 0.7 -> 0.986397 (best), 1.0 -> 0.987757. Longer LR decay helps in a 5-min run; optimum ~0.7.
- `#27` **a0** [commit `67cd043`]: Value embeddings on ALL 8 layers instead of every other layer: val_bpb 0.986397 -> 0.984833. VE are embedding lookups, so they add capacity at almost zero FLOP cost (1840->1817 steps, 476M tokens) - exactly the right trade when compute-bound. New best 0.984833.
- `#37` **a0** [commit `6fc32ab`]: MLP expansion sweep at dim 512 / depth 8 (everything else = commit 67cd043): 3x -> 0.990728, 4x -> 0.984833, 5x -> 0.984049, 6x -> 0.983147. Monotone improvement well past the standard 4x; the model is starved of FFN capacity, and the lost steps (1801 -> ~1650) are worth it. Continuing up.
- `#46` **a0** [commit `156f18a`]: With the 6x MLP the Muon LR optimum moves down: MATRIX_LR 0.04 -> 0.983147, 0.03 -> 0.982289 (best), 0.022 -> 0.983141. New best 0.982289 = batch 2^18 + short window 512 + SSL pattern + VE on all layers + MLP 6x + MATRIX_LR 0.03 + WARMDOWN 0.7.
- `#54` **a0** [commit `49d825d`]: Logit softcap matters a lot: raising it 15 -> 30 costs 0.0035 bpb (0.982289 -> 0.985770). It is acting as real regularisation, not a safety clamp. Probing tighter next.
- `#59` **a0** [commit `a2d0782`]: Attention softmax_scale with QK-norm: default 1/sqrt(128)=0.088 -> 0.12 improves 0.982289 -> 0.981856 (new best); 0.17 is much worse (0.985007). Since q,k are rms-normed the default 1/sqrt(head_dim) is not the right temperature; ~0.12 (the nanochat value) is near the peak.
- `#64` **a0** [commit `0ad512b`]: FINAL (a0): best val_bpb 0.980742 at commit 0ad512b, vs 0.997359 baseline (-0.0166). Config = shipped train.py with: TOTAL_BATCH_SIZE 2^18 (grad_accum 1, DEVICE_BATCH_SIZE 128); short sliding window = sequence_len//4 = 512 and WINDOW_PATTERN 'SSL'; has_ve() -> True (value embeddings on all 8 layers); MLP expansion 6x; MATRIX_LR 0.03; WARMDOWN_RATIO 0.7; fa3 softmax_scale=0.12; EMBEDDING_LR 0.32. EMBEDDING_LR was still improving monotonically at the last run (0.9->0.983091, 0.6->0.981856, 0.45->0.981181, 0.32->0.980742), so that axis is not exhausted.

## Disconfirmations (negative results, attempts to falsify)

- `#4` **a0** [commit `a2b0086`]: DEPTH=10 (dim 640, 5 heads x 128) gives val_bpb 1.002094 vs baseline 0.997359 (DEPTH=8, dim 512). Bigger model is NOT better at the 5-min budget: 569 steps / 298M tokens at 42.3% MFU vs ~1000 steps for depth 8. Also note for everyone: vocab_size is 8192 (not 32768 as the GPTConfig default suggests), so the lm_head is cheap (5.2M params) and the transformer body dominates FLOPs.
- `#6` **a0** [commit `ce452c5`]: HEAD_DIM 128->64 at DEPTH=8/dim=512 (8 heads x 64 instead of 4 x 128) is worse on BOTH axes: val_bpb 1.004693 vs 0.997359 baseline, and MFU drops 42.3%->37.15% (FA3 is slower at head_dim 64). 890 steps / 466M tokens. Do not split heads finer.
- `#10` **a0** [commit `88efe7e`]: TOTAL_BATCH_SIZE 2^17 / DEVICE_BATCH_SIZE 64: val_bpb 0.998166, worse than 2^18 (0.989498). Throughput falls off a cliff at device batch 64 (MFU 32.7% vs 37.3%, 407M vs 466M tokens). 2^18 with device batch 128 is an interior optimum on the batch-size axis.
- `#12` **a0** [commit `c54a1a1`]: MATRIX_LR 0.04 -> 0.055 at batch 2^18: val_bpb 0.993275, worse than 0.989498. Muon LR 0.04 is at/near optimum; the batch-size win is not a disguised LR win.
- `#14` **a0** [commit `f4c9912`]: U-net LIFO skip connections between layer halves (learned scalar skip_lambdas init 0.1, in the x0 param group) HURT: val_bpb 0.996679 vs 0.989498 for the same config without them. The existing x0_lambdas embedding-shortcut already covers this; reverted.
- `#18` **a0** [commit `46ebd9a`]: Short window 256 (long_window//8): val_bpb 0.988849, worse than 512 (0.987884). 512 is the optimum on that axis.
- `#29` **a0** [commit `a677bf9`]: Token-smear at the embedding (x += learned scalar * previous-token embedding, init 0.3, before x0 is taken): val_bpb 0.988392 vs 0.984833 without. Hurts here, probably because per-layer value embeddings already inject token identity. Also +7GB VRAM from the shift copy.
- `#32` **a0** [commit `209a3d8`]: torch._inductor.config.coordinate_descent_tuning=True is a net loss: MFU 36.30% -> 35.79% (472M -> 465M tokens), val_bpb 0.984833 -> 0.985614, and +90s of compile (total_seconds 345 -> 435, uncomfortably close to the 600s arena timeout). Inductor's default kernels are already good here; compiler-side throughput tuning is a dead end for this model.
- `#34` **a0** [commit `1e6b010`]: MLP expansion 4x -> 3x: val_bpb 0.990728 vs 0.984833. Big loss. MLP capacity is not the place to buy FLOPs back; the extra steps do not compensate.
- `#39` **a0** [commit `4a8e88c`]: MLP expansion 8x: val_bpb 0.983318, no better than 6x (0.983147). The MLP-width curve plateaus at ~6x; 6x is the pick (also 8GB less VRAM).
- `#41` **a0** [commit `6f973e4`]: Gated relu^2 MLP (c_fc -> 2*hidden, x = relu(a)^2 * b, hidden 4x so params/FLOPs match the ungated 6x MLP): val_bpb 0.989912 vs 0.983147 for plain ungated relu^2 at 6x. Gating is a clear loss here at matched cost.
- `#43` **a0** [commit `15fcdd9`]: DEPTH 8 -> 10 at fixed dim 512 (ASPECT_RATIO 51) with the 6x MLP: val_bpb 0.983005 vs 0.983147 at depth 8 - inside noise (~0.0003), and costs 20GB more VRAM and ~15% of the steps. Depth is flat in 8..10 once the MLP is wide; capacity is better bought in the FFN.
- `#49` **a0** [commit `4e8d329`]: Muon WEIGHT_DECAY sweep at the 6x-MLP/LR-0.03 config: 0.1 -> 0.983402, 0.2 -> 0.982289 (shipped value, best), 0.4 -> 0.984308. WD 0.2 is already optimal.
- `#51` **a0** [commit `b8017b1`]: MLP 8x re-probed at the lower MATRIX_LR 0.03: 0.982867 vs 0.982289 at 6x. 6x remains the optimum; the earlier 6x-vs-8x plateau was not an LR artifact.
- `#56` **a0** [commit `52e3cc9`]: Logit softcap 10: 0.982384, indistinguishable from 15 (0.982289). The softcap curve is flat below 15 and sharply worse above it; keep 15.
- `#61` **a0** [commit `cc6d42a`]: WARMDOWN_RATIO 0.85 at the best config: 0.982201 vs 0.981856 at 0.7. 0.7 confirmed as the warmdown optimum (0.5/0.7/0.85/1.0 = 0.987388(old cfg)/best/0.982201/worse).

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

