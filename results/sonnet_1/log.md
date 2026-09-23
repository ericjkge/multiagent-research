# Shared research log

Baseline at start of the cell: **val_bpb 1.014027**

## Round 0
_Baseline entering the round: val_bpb 1.014027 (commit 228791f)_

### Proposals
- `#2` **a0 — Sweep model depth (DEPTH=8 → 10)**
  - why: At fixed training FLOPs/time budget, small transformers are typically more sample-efficient per-token when made modestly deeper rather than wider, and depth is completely unexplored in the log so far.
  - detail: Single-line change: set `DEPTH = 10` (from 8) in the hyperparameters block, leaving `ASPECT_RATIO=64` and `HEAD_DIM=128` unchanged. This raises base_dim to 640, which rounds up to model_dim=640 (5 heads at head_dim=128) via `build_model_config`. Everything else (batch size, LRs, schedules, window pattern) stays as-is since `setup_optimizer`'s `dmodel_lr_scale` already adjusts AdamW LRs for the new model_dim automatically. This isolates the depth/width tradeoff as the only variable. If this improves val_bpb, a natural follow-up is a small grid around depth (e.g. 9, 11) to find the local optimum; if it regresses, it suggests the original DEPTH=8 choice was already near-optimal for this 5-min budget and the log should note not to repeat depth sweeps.</detail>
<parameter name="notes">Given this script is already heavily tuned (Muon+NorMuon, value embeddings, sliding window pattern, polar-express orthogonalization) it looks like it derives from a well-optimized nanochat/modded-nanogpt lineage — architecture-level micro-tweaks may have already been tried by the original author before this cell started, so depth is one of the few large untested levers left.

### Experiments run
- `a0/v0` val_bpb **1.002279** (66.5 GB) — Changed DEPTH from 8 to 10 (model_dim goes 512->640, 4->5 heads), everything else unchanged.
- `a0/v1` val_bpb **1.002351** (66.5 GB) — Set DEPTH=10 (from 8), keeping ASPECT_RATIO=64 and HEAD_DIM=128; model_dim becomes 640 (5 heads), all LR/schedule logic unchanged since dmodel_lr_scale auto-adjusts.
- `a0/v2` val_bpb **1.002712** (66.5 GB) — Changed DEPTH from 8 to 10 (model_dim goes 512->640, 4->5 heads); everything else unchanged, dmodel_lr_scale auto-adjusts AdamW LRs.
- `a0/v3` val_bpb **1.002703** (66.5 GB) — Changed DEPTH from 8 to 10 (model_dim auto-scales to 640, 5 heads); no other hyperparameters touched.
- `a0/v4` val_bpb **1.002496** (66.5 GB) — Changed DEPTH from 8 to 10 (model_dim auto-adjusts from 512 to 640, 5 heads); everything else unchanged.

**Outcome:** a0/v0 won; new baseline val_bpb 1.002279.

## Round 1
_Baseline entering the round: val_bpb 1.002279 (commit 9acd136)_

### Proposals
- `#11` **a0 — Extend depth sweep: DEPTH=10 → 12**
  - why: Depth 8→10 gave a clear, consistent win (~1.2% bpb reduction across 5 seeds), so the depth-vs-width tradeoff curve for this time budget likely hasn't peaked yet at 10.
  - detail: Set `DEPTH = 12` (from 10) in the hyperparameters block; base_dim becomes 768, rounding to model_dim=768 (6 heads at head_dim=128) via `build_model_config`. Everything else (ASPECT_RATIO=64, HEAD_DIM=128, batch size, LR schedule) stays unchanged since `dmodel_lr_scale` auto-adjusts AdamW LRs. Watch peak VRAM (was 66.5GB at depth=10) — if it's near the GPU limit, drop `DEVICE_BATCH_SIZE` from 128 to 96 or 64 (compensating with more grad_accum_steps so TOTAL_BATCH_SIZE stays fixed) to keep memory headroom.</detail>
<parameter name="notes">If depth=12 regresses vs depth=10, that pins the local optimum near 10-11 for this 5-min/token budget and further depth sweeps should stop; a finer step (DEPTH=11) would be the next probe instead of jumping to 14.

### Experiments run
- `a0/v0` val_bpb **1.016037** (47.8 GB) — Set DEPTH=12 (model_dim 640->768, 6 heads); also reduced DEVICE_BATCH_SIZE 128->64 (more grad_accum_steps) to stay within VRAM headroom.
- `a0/v1` val_bpb **1.016254** (47.8 GB) — Set DEPTH=12 (model_dim 768, 6 heads); reduced DEVICE_BATCH_SIZE 128->64 (grad_accum doubled 2->4) to keep VRAM headroom vs depth=10's 66.5GB.
- `a0/v2` **CRASH** — Set DEPTH=12 (model_dim 640->768, 6 heads); at DEVICE_BATCH_SIZE=128 (matching depth=10 setup) this OOMs, so DEVICE_BATCH_SIZE was reduced to 64 (grad_accum 2->4) to fit memory. (File "/root/autoresearch/.venv/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 1130, in forward
    return compiled_fn(full_args)
  File "/ [...])
- `a0/v3` val_bpb **1.016225** (47.8 GB) — Set DEPTH=12 (model_dim=768, 6 heads) and reduced DEVICE_BATCH_SIZE from 128 to 64 (grad_accum_steps 2->8) to keep VRAM under budget.
- `a0/v4` val_bpb **1.015935** (47.8 GB) — Set DEPTH=12 (from 10); OOM'd at DEVICE_BATCH_SIZE=128 so reduced to 64 (grad_accum 2->4, TOTAL_BATCH_SIZE unchanged) to fit in 79GB VRAM.

**Outcome:** nothing beat the baseline; it is unchanged at val_bpb 1.002279.

## Round 2
_Baseline entering the round: val_bpb 1.002279 (commit 9acd136)_

### Proposals
- `#20` **a0 — Tune Muon matrix LR (MATRIX_LR sweep)**
  - why: Muon drives updates for the vast majority of parameters (all transformer matrices) and its LR has never been touched in this log — two rounds of depth sweeps found a local optimum on architecture, but optimizer LR is a completely untested, potentially higher-leverage lever.
  - detail: Starting from current baseline (DEPTH=10, all else unchanged), change only `MATRIX_LR = 0.04` to `MATRIX_LR = 0.05` (a 25% increase). Everything else (ASPECT_RATIO, HEAD_DIM, DEVICE_BATCH_SIZE=128, EMBEDDING_LR, UNEMBEDDING_LR, SCALAR_LR, WEIGHT_DECAY, schedules) stays fixed so the LR change is isolated. This directly scales the `lr` passed into Muon param groups in `setup_optimizer`, which flows into `muon_step_fused` via `group["lr"] * max(1.0, shape[-2]/shape[-1])**0.5`. If 0.05 improves val_bpb, a natural follow-up is 0.06 or a symmetric check at 0.03 to map the curve; if it regresses, try 0.03 instead (lower LR) since we won't know a priori which direction is better without a first data point.</detail>
<parameter name="notes">If this round's other agent proposals also touch depth/width, I'd rather defer to LR since architecture search already used 10 runs with a fairly clear local optimum at DEPTH=10, VRAM≈66.5GB, batch=128.

### Experiments run
- `a0/v0` val_bpb **1.005825** (66.5 GB) — Changed MATRIX_LR from 0.04 to 0.05 (Muon LR for transformer matrix params), all else unchanged from DEPTH=10 baseline.
- `a0/v1` val_bpb **1.006023** (66.5 GB) — Raised MATRIX_LR from 0.04 to 0.05 (25% increase), all else unchanged from DEPTH=10 baseline.
- `a0/v2` val_bpb **1.005816** (66.5 GB) — Increased MATRIX_LR from 0.04 to 0.05 (25% higher Muon LR), all else unchanged from DEPTH=10 baseline.
- `a0/v3` val_bpb **1.005762** (66.5 GB) — Increased MATRIX_LR from 0.04 to 0.05 (25% higher Muon LR), all else unchanged at DEPTH=10 baseline.
- `a0/v4` val_bpb **1.005993** (66.5 GB) — Increased MATRIX_LR from 0.04 to 0.05 (25% higher Muon LR for transformer matrix params), all else unchanged from DEPTH=10 baseline.

**Outcome:** nothing beat the baseline; it is unchanged at val_bpb 1.002279.

## Round 3
_Baseline entering the round: val_bpb 1.002279 (commit 9acd136)_

### Proposals
- `#29` **a0 — Sweep LR warmdown length (WARMDOWN_RATIO 0.5 → 0.8)**
  - why: Two rounds of architecture/LR-magnitude sweeps found local optima and failed to improve further, but the LR *schedule shape* itself has never been touched despite this being a fixed-time-budget run where warmdown length strongly affects how well the model settles.
  - detail: In the hyperparameters block, change `WARMDOWN_RATIO = 0.5` to `WARMDOWN_RATIO = 0.8` (from 0.5), keeping everything else (DEPTH=10, MATRIX_LR=0.04, etc.) at current baseline values. This changes `get_lr_multiplier` so the linear LR decay to FINAL_LR_FRAC=0.0 begins at progress=0.2 instead of progress=0.5, giving the model a longer, gentler cooldown within the same fixed 5-minute budget — a completely untested axis (only model size and Muon LR magnitude have been swept so far). If 0.8 improves val_bpb, a natural follow-up is 1.0 (cooldown for the entire run, no flat-LR phase) or 0.65; if it regresses, try shortening to 0.3 instead, since the sign of the effect isn't known yet.

### Experiments run
- `a0/v0` val_bpb **1.004892** (66.5 GB) — Changed WARMDOWN_RATIO from 0.5 to 0.8, giving a longer/gentler linear LR cooldown starting at progress=0.2 instead of 0.5, all else unchanged from DEPTH=10 baseline.
- `a0/v1` val_bpb **1.005098** (66.5 GB) — Changed WARMDOWN_RATIO from 0.5 to 0.8 (longer, gentler LR cooldown starting at progress=0.2 instead of 0.5), all else unchanged from DEPTH=10 baseline.
- `a0/v2` val_bpb **1.004858** (66.5 GB) — Increased WARMDOWN_RATIO from 0.5 to 0.8 (longer, gentler linear cooldown starting at progress=0.2 instead of 0.5), all else unchanged from DEPTH=10 baseline.
- `a0/v3` val_bpb **1.005063** (66.5 GB) — Changed WARMDOWN_RATIO from 0.5 to 0.8, giving a longer/gentler linear LR cooldown (starts at progress=0.2 instead of 0.5) within the same fixed 5-minute budget; all else unchanged from DEPTH=10 basel [...]
- `a0/v4` val_bpb **1.005052** (66.5 GB) — Changed WARMDOWN_RATIO from 0.5 to 0.8, giving a longer/gentler linear LR cooldown starting at progress=0.2 instead of 0.5, all else unchanged from DEPTH=10 baseline.

**Outcome:** nothing beat the baseline; it is unchanged at val_bpb 1.002279.

## Round 4
_Baseline entering the round: val_bpb 1.002279 (commit 9acd136)_

### Proposals
- `#38` **a0 — Matrix LR down-sweep (MATRIX_LR 0.04 → 0.03)**
  - why: Round 2 only tested raising MATRIX_LR (which hurt); the log explicitly flagged lowering it as the untested complementary direction, and three consecutive up/increase-style sweeps have all failed, suggesting the baseline may sit above rather than below its optimum on this axis.
  - detail: From the current baseline (DEPTH=10, all else unchanged), change only `MATRIX_LR = 0.04` to `MATRIX_LR = 0.03` (a 25% decrease) in the hyperparameters block. This flows into `setup_optimizer`'s Muon param groups (`lr=matrix_lr`) and then into `muon_step_fused` via `group["lr"] * max(1.0, shape[-2]/shape[-1])**0.5`. Nothing else changes (ASPECT_RATIO, HEAD_DIM, DEVICE_BATCH_SIZE=128, WARMDOWN_RATIO=0.5, EMBEDDING_LR, etc.) so the LR magnitude is the sole isolated variable, directly mirroring the round-2 methodology but in the untested direction.

### Experiments run
- `a0/v0` val_bpb **1.006195** (66.5 GB) — Decreased MATRIX_LR (Muon LR) from 0.04 to 0.03, 25% lower, all else unchanged from DEPTH=10 baseline.
- `a0/v1` val_bpb **1.005856** (66.5 GB) — Decreased MATRIX_LR from 0.04 to 0.03 (Muon LR for transformer matrix params), all else unchanged from DEPTH=10 baseline.
- `a0/v2` val_bpb **1.005875** (66.5 GB) — Lowered MATRIX_LR from 0.04 to 0.03 (25% decrease), all else unchanged from DEPTH=10 baseline.
- `a0/v3` val_bpb **1.005651** (66.5 GB) — Decreased MATRIX_LR from 0.04 to 0.03 (25% lower Muon LR for transformer matrix params), all else unchanged from DEPTH=10 baseline.
- `a0/v4` val_bpb **1.005562** (66.5 GB) — Decreased MATRIX_LR from 0.04 to 0.03 (25% lower Muon LR), all else unchanged from DEPTH=10 baseline.

**Outcome:** nothing beat the baseline; it is unchanged at val_bpb 1.002279.

## Round 5
_Baseline entering the round: val_bpb 1.002279 (commit 9acd136)_

### Proposals
- `#47` **a0 — Lower weight decay for short training budget (WEIGHT_DECAY 0.2 → 0.1)**
  - why: Four rounds of LR-magnitude/schedule sweeps near DEPTH=10 all failed; weight decay is untested, and strong decay (0.2) may over-shrink weights in this short, few-thousand-step regime.
  - detail: In the hyperparameters block, change `WEIGHT_DECAY = 0.2` to `WEIGHT_DECAY = 0.1` (from current DEPTH=10 baseline, everything else unchanged: ASPECT_RATIO=64, HEAD_DIM=128, MATRIX_LR=0.04, WARMDOWN_RATIO=0.5, DEVICE_BATCH_SIZE=128). This flows into `setup_optimizer`'s Muon param groups (`weight_decay=weight_decay`) and then `get_weight_decay(progress) = WEIGHT_DECAY * (1 - progress)`, which is applied each step in `muon_step_fused` as cautious decay (`lr * wd * stacked_params * mask`). Halving the base value halves the decay pressure throughout training while preserving the same linear anneal-to-zero shape. If this improves val_bpb, a natural follow-up is WEIGHT_DECAY=0.05 or 0.0 (no decay); if it regresses, the untested direction (raising to 0.3) becomes the next candidate.

### Experiments run
- `a0/v0` val_bpb **1.003733** (66.5 GB) — Changed WEIGHT_DECAY from 0.2 to 0.1 (halved cautious Muon weight decay), all else unchanged from DEPTH=10 baseline.
- `a0/v1` val_bpb **1.004031** (66.5 GB) — Halved WEIGHT_DECAY from 0.2 to 0.1 (cautious Muon decay), all else unchanged from DEPTH=10 baseline.
- `a0/v2` val_bpb **1.003496** (66.5 GB) — Halved WEIGHT_DECAY from 0.2 to 0.1, all else unchanged from DEPTH=10 baseline.
- `a0/v3` val_bpb **1.003675** (66.5 GB) — Halved cautious Muon weight decay from 0.2 to 0.1, everything else unchanged from DEPTH=10 baseline.
- `a0/v4` val_bpb **1.003940** (66.5 GB) — Halved cautious Muon weight decay from 0.2 to 0.1, all else unchanged from DEPTH=10 baseline.

**Outcome:** nothing beat the baseline; it is unchanged at val_bpb 1.002279.

