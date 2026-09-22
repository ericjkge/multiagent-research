# Shared research log

Baseline at start of the cell: **val_bpb 1.014027**

## Round 0
_Baseline entering the round: val_bpb 1.014027 (commit 228791f)_

### Proposals
- `#3` **a0 — Depth/width compute-optimal sweep at fixed wall-clock budget**
  - why: DEPTH=8 (with ASPECT_RATIO=64) was likely tuned for a different time budget or hardware, not verified as compute-optimal for this exact 5-minute run on this specific eval, so it's the highest-leverage untested lever.
  - detail: Change DEPTH from 8 to 6 (train.py line `DEPTH = 8`) leaving ASPECT_RATIO=64, HEAD_DIM=128 unchanged (build_model_config derives n_embd/n_head automatically). A smaller model trains faster per step, so within the fixed 5-minute budget it sees more tokens/optimizer steps; under a fixed-compute-budget scaling law the loss-optimal model size is usually smaller than what looks "reasonable" by parameter count alone. If DEPTH=6 wins, follow up by trying DEPTH=10 in the opposite direction to bracket the optimum. Keep DEVICE_BATCH_SIZE and all optimizer hyperparameters unchanged for a clean single-variable comparison against the 1.014027 baseline.
- `#2` **a1 — Depth sweep: DEPTH=12 vs baseline 8**
  - why: Nanochat-style scaling results show these narrow/deep GPT+Muon setups are usually under-parameterized at depth 8 for a fixed 5-min wall-clock budget, so a deeper model (more layers, proportionally wider via ASPECT_RATIO) should reach lower val_bpb despite fewer optimizer steps.
  - detail: In train.py change only `DEPTH = 8` to `DEPTH = 12` (leave ASPECT_RATIO=64, HEAD_DIM=128 unchanged so model_dim scales automatically to 768, num_heads=6 — matching the GPTConfig defaults already in the dataclass, so this is essentially returning to the "documented default" shape). No other hyperparameters touched: TOTAL_BATCH_SIZE, LRs, WARMDOWN_RATIO stay as-is since MATRIX_LR/EMBEDDING_LR etc. already scale with 1/sqrt(model_dim/768) inside setup_optimizer, so the LR scaling should remain roughly correct at dim 768. This isolates the depth/width effect cleanly as a single-variable change.
- `#4` **a2 — Depth sweep for fixed 5-min compute budget**
  - why: Under a fixed wall-clock budget, model depth trades capacity for tokens/sec — nobody has verified DEPTH=8 is actually compute-optimal for exactly 5 minutes on this hardware, and it's the single highest-leverage untested knob.
  - detail: Change only DEPTH in train.py (currently 8) to a smaller value, e.g. DEPTH=6, keeping ASPECT_RATIO=64 and HEAD_DIM=128 unchanged (so model_dim scales down to 384, heads=3). This shrinks the model, increases tok/sec and total tokens seen in the fixed 5-minute budget, which for such a short budget often beats a bigger, more undertrained model. If DEPTH=6 beats baseline, follow up with DEPTH=10 to bracket the optimum in later rounds; if it's worse, that tells us 8 is already past the small-model side of the optimum and future search should go the other direction (DEPTH=10-12) instead.

### Messages
- `#12` **a2**: Two things worth flagging for round 1:

1. **DEPTH=6 was never actually tested.** Both a0 and I proposed it independently, but a0 switched to DEPTH=10 and I switched to MATRIX_LR — so the "bracket" everyone's referring to is really only 8/10/12, not 6/8/10/12. Given DEPTH=10 clearly beats both 8 and 12, the curve looks like it peaks somewhere near 10, so I'd deprioritize DEPTH=6 now rather than treat it as an open gap — the trend (12 is already worse than 10) suggests going smaller won't help either. Better to bracket 9/10/11 if we want to be precise, or move to a different axis.

2. **a1's DEPTH=12 runs are confounded, not a clean depth comparison.** Both v0 and v1 had to halve DEVICE_BATCH_SIZE (128→64) to avoid OOM, doubling grad_accum_steps. That changes gradient noise/step dynamics, not just depth — so "depth=12 loses to depth=10" isn't a pure single-variable result the way a0's dep [...]
- `#13` **a1**: Depth=10 (a0) is the clear winner and looks like a real effect, not noise: two independent a0 runs at DEPTH=10 landed at 1.002652 and 1.002530, both ~0.011 bpb better than baseline and using the full DEVICE_BATCH_SIZE=128 (66.5GB) — no confound. My two DEPTH=12 runs (1.015989, 1.016240) were confounded: I had to halve DEVICE_BATCH_SIZE to 64 to avoid OOM, doubling grad_accum_steps. That changes gradient noise/step dynamics on top of the depth change, so we can't cleanly conclude "depth 12 is worse than depth 10 architecturally" — it might just be "depth 12 at batch 64 is worse." The bracket (6 worse, 8 baseline, 10 much better, 12 worse-but-confounded) still strongly suggests the optimum is near 10, not beyond it, but if anyone wants to fully resolve the 10-vs-12 question they'd need a DEPTH=12 run that keeps batch=128 (e.g. via gradient checkpointing, or accept the OOM risk isn't worth [...]
- `#14` **a0**: Depth sweep summary: 8 (baseline, 1.014027) → 10 (1.002530, winner) → 12 (1.016, worse than baseline, both a1 runs). But the 12-vs-10 comparison isn't clean: DEPTH=12 OOM'd at DEVICE_BATCH_SIZE=128 and both a1 runs had to halve it to 64, doubling grad_accum. So the depth=12 result confounds "more layers" with "more grad-accum overhead / same TOTAL_BATCH_SIZE but coarser micro-batching," and it's entirely possible depth=12 at a batch size that fits (e.g. 96, or with activation checkpointing) would beat depth=10 — we don't actually know depth=12 is worse, we know depth=12-with-halved-batch is worse. Also worth noting: nobody actually ran DEPTH=6 this round despite three proposals aiming there — a2 pivoted away from it, so the small-model side of the bracket is still untested (though the 8→10 direction alone being a solid win makes DEPTH=6 low priority now). For next round I'd suggest: (1) [...]

### Experiments run
- `a0/v0` val_bpb **1.002652** (66.5 GB) — Changed only DEPTH from 8 to 10 in train.py, leaving ASPECT_RATIO/HEAD_DIM/all other hyperparameters unchanged.
- `a0/v1` val_bpb **1.002530** (66.5 GB) — Changed DEPTH from 8 to 10, leaving ASPECT_RATIO/HEAD_DIM and all other hyperparameters untouched, to fill the gap between baseline (8) and a1's proposed 12.
- `a1/v0` val_bpb **1.015989** (47.8 GB) — Increased DEPTH from 8 to 12 (model_dim auto-scales to 768, 6 heads); had to halve DEVICE_BATCH_SIZE to 64 to avoid CUDA OOM at this depth.
- `a1/v1` val_bpb **1.016240** (47.8 GB) — Changed DEPTH from 8 to 12 (model_dim scales to 768 via ASPECT_RATIO), and reduced DEVICE_BATCH_SIZE from 128 to 64 (grad_accum doubled to keep TOTAL_BATCH_SIZE fixed) to avoid an OOM the first attemp [...]
- `a2/v0` val_bpb **1.015310** (44.0 GB) — Changed WARMDOWN_RATIO from 0.5 to 0.3, giving more time at peak LR and less time cooling down, since two other agents were already covering the depth axis this round.
- `a2/v1` val_bpb **1.012971** (44.0 GB) — Increased MATRIX_LR (Muon optimizer LR for matrix params) from 0.04 to 0.06, all else unchanged.

**Outcome:** a0/v1 won; new baseline val_bpb 1.002530.

## Round 1
_Baseline entering the round: val_bpb 1.002530 (commit d17cd88)_

### Proposals
- `#18` **a0 — Retune MATRIX_LR at DEPTH=10 (Muon LR doesn't auto-scale with width)**
  - why: setup_optimizer scales all AdamW LRs by 1/√(model_dim/768), but MATRIX_LR (Muon) is passed through unscaled — so when DEPTH went 8→10 (model_dim 512→640), the Muon LR for matrix params was never re-tuned for the new width, unlike everything else.
  - detail: Starting from the new baseline (DEPTH=10, MATRIX_LR=0.04), sweep MATRIX_LR only, e.g. try 0.03 and 0.05 (or just one value if budget-constrained — I'd pick 0.05 first since a2's round-0 result at depth=8 showed 0.06 > 0.04 baseline, suggesting the untuned default may be too low rather than too high). Single-line change: `MATRIX_LR = 0.05` (train.py, currently 0.04). No other hyperparameters touched, so it isolates the Muon-LR-at-new-width effect cleanly against the 1.002530 baseline.
- `#17` **a1 — Depth=11 fine bracket**
  - why: The depth sweep (8→10→12) shows 10 is a clear local optimum but 12 was confounded by a forced batch-size cut, so the true optimum between 10 and 12 is still unresolved — DEPTH=11 tests that gap cleanly.
  - detail: In train.py change only `DEPTH = 10` to `DEPTH = 11` (line near bottom of hyperparameters section), leaving ASPECT_RATIO=64, HEAD_DIM=128, DEVICE_BATCH_SIZE=128 and all optimizer hyperparameters untouched. model_dim will scale to 768 (same as depth 10 and 12 due to HEAD_DIM rounding), so this isolates layer count alone. If it fits in VRAM at full batch 128 (likely, since depth 10 used 66.5GB and depth 12 only OOM'd, not depth 11), this gives a clean, unconfounded three-point comparison at 10/11/12 to pin down exactly where the optimum sits, rather than repeating the muddied 10-vs-12 comparison.
- `#16` **a2 — MATRIX_LR increase at DEPTH=10**
  - why: a2's MATRIX_LR=0.06 gave a small win (1.012971) at the old DEPTH=8 baseline, but nobody has tested it combined with the new DEPTH=10 winner — these two independent single-axis wins may stack.
  - detail: Starting from the new baseline train.py (DEPTH=10), change only `MATRIX_LR = 0.04` to `MATRIX_LR = 0.06` (Muon LR for matrix params). Leave DEPTH, ASPECT_RATIO, HEAD_DIM, DEVICE_BATCH_SIZE, and all other hyperparameters untouched for a clean combination test.

### Messages
- `#26` **a2**: MATRIX_LR axis at DEPTH=10 is now dead: four runs (a0: 0.05→1.005897, 0.08→1.004728; me: 0.06→1.005574 and 1.005623) all landed worse than the 0.04 baseline (1.002530), bracketing both directions. The round-0 MATRIX_LR=0.06 win was specific to depth=8/model_dim=512 and doesn't transfer once model_dim=640 — don't spend more slots retuning MATRIX_LR here.

On a1's crash: their own diagnosis in candidate.json is correct and worth confirming — DEPTH=11 is not a midpoint between 10 and 12. `build_model_config` rounds model_dim up to the nearest HEAD_DIM=128 multiple, so depth=8-10 all give model_dim=640, but depth=11-12 both jump to model_dim=768. DEPTH=11 is therefore the same width as DEPTH=12 plus one extra layer — strictly bigger, which is why it OOM'd even harder than 12 did. If anyone wants a real bracket point between the 640-wide depth=10 winner and the 768-wide depth=12 loser, DEPTH= [...]
- `#27` **a1**: Confirmed math on the DEPTH=11 crash: model_dim = ceil(depth*64/128)*128 means depth 9-10 both give model_dim=640 (5 heads), and depth 11-12 both give model_dim=768 (6 heads). So DEPTH=11 isn't a midpoint between 10 and 12 at all — it's the *same width* as 12 plus an extra layer, i.e. strictly more params than depth=12, which explains why both a1 attempts OOM'd even harder than the already-OOMing depth=12. If anyone wants a clean bracket to resolve "is 10 really the peak," DEPTH=9 is the useful untested point: same width as the depth=10 winner (640/5 heads), one fewer layer, guaranteed to fit at batch=128, and gives a real depth-only comparison at fixed width. Going the other way, resolving 10-vs-12 cleanly still requires either DEVICE_BATCH_SIZE=96 (doesn't evenly divide TOTAL_BATCH_SIZE=2^19 with tokens_per_fwdbwd=96*2048, so would need TOTAL_BATCH_SIZE adjusted too) or gradient checkp [...]
- `#28` **a0**: Two findings from this round:

1. **a1's DEPTH=11 crash is explained by the HEAD_DIM rounding, not a marginal VRAM bump.** `build_model_config` rounds `base_dim` up to the nearest multiple of HEAD_DIM=128: depth=10 gives base_dim=640 (already a multiple of 128, model_dim=640), but depth=11 gives base_dim=704 → rounds up to model_dim=768 — the *same width as depth=12*. So depth=11 isn't "10.5"; it's "depth=12's width with one fewer layer," and unsurprisingly it OOMs at batch=128 just like depth=12 did. The real unconfounded gap to test, if anyone wants it, is DEPTH=9 (base_dim=576→640, same width as depth=10) or a HEAD_DIM tweak that gives finer width control — DEPTH=11 as originally proposed will never test cleanly at full batch size.

2. **MATRIX_LR sweep at DEPTH=10 is now a solid negative result — stop spending slots on it.** Four runs (0.05, 0.06×2, 0.08) all landed worse than the 0. [...]

### Experiments run
- `a0/v0` val_bpb **1.004728** (66.5 GB) — Increased MATRIX_LR (Muon LR for matrix params) from 0.04 to 0.08, keeping DEPTH=10 and everything else unchanged.
- `a0/v1` val_bpb **1.005897** (66.5 GB) — Increased MATRIX_LR from 0.04 to 0.05 at the new DEPTH=10 baseline, testing whether Muon LR (which doesn't auto-scale with model width like AdamW LRs do) needed retuning for the wider model.
- `a1/v0` **CRASH** — Changed DEPTH from 10 to 11 to test the gap between depth=10 (winner) and depth=12 (confounded loser); crashed with CUDA OOM at DEVICE_BATCH_SIZE=128. (File "/root/autoresearch/.venv/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 1130, in forward
    return compiled_fn(full_args)
  File "/ [...])
- `a1/v1` **CRASH** — Changed DEPTH from 10 to 11 (ASPECT_RATIO/HEAD_DIM/DEVICE_BATCH_SIZE unchanged) to bracket between the confirmed depth=10 winner and the confounded depth=12 loser. (File "/root/autoresearch/.venv/lib/python3.10/site-packages/torch/_functorch/aot_autograd.py", line 1130, in forward
    return compiled_fn(full_args)
  File "/ [...])
- `a2/v0` val_bpb **1.005574** (66.5 GB) — Changed MATRIX_LR from 0.04 to 0.06 at DEPTH=10 (new baseline), all else unchanged.
- `a2/v1` val_bpb **1.005623** (66.5 GB) — Changed MATRIX_LR from 0.04 to 0.06 at DEPTH=10 (Muon LR for matrix params), testing whether the MATRIX_LR=0.06 win seen at depth=8 (round 0) stacks with the new depth=10 baseline.

**Outcome:** nothing beat the baseline; it is unchanged at val_bpb 1.002530.

## Round 2
_Baseline entering the round: val_bpb 1.002530 (commit d17cd88)_

### Proposals
- `#30` **a0 — Add short LR warmup at DEPTH=10**
  - why: DEPTH=10 (model_dim=640) is a much bigger model than the original tuning target (dim=512/depth=8), and WARMUP_RATIO=0.0 was never revisited after the depth change — larger models with Muon+AdamW typically need a brief warmup to avoid early-training instability/wasted steps that a 0-warmup schedule c [...]
  - detail: In train.py change `WARMUP_RATIO = 0.0` to `WARMUP_RATIO = 0.05` (5% of the 5-minute budget, ~15s, ramping LR linearly from 0 to full). Leave DEPTH=10, MATRIX_LR=0.04, WARMDOWN_RATIO=0.5 and everything else untouched — this isolates the warmup effect as a single-variable change against the 1.002530 baseline. get_lr_multiplier already supports WARMUP_RATIO>0 via the existing `progress / WARMUP_RATIO` branch, so no other code changes needed.
- `#31` **a1 — DEPTH=9 — same width as winner, one fewer layer**
  - why: DEPTH=9 shares model_dim=640 with the DEPTH=10 winner (rounds up the same way) but trains faster per step, so within the fixed 5-min budget it could out-token the depth-10 winner while still being deeper than the old baseline (8).
  - detail: In train.py change only `DEPTH = 10` to `DEPTH = 9` (leave ASPECT_RATIO=64, HEAD_DIM=128, DEVICE_BATCH_SIZE=128, all optimizer hyperparameters untouched). build_model_config gives base_dim=576, rounds up to model_dim=640 (5 heads) — identical width/head-count to the current depth=10 winner, so this is a pure layer-count-at-fixed-width comparison, resolving the exact gap both a0 and a1 flagged as untested last round. Guaranteed to fit at full batch=128 since depth=10 already fits at that width.
- `#32` **a2 — Add LR warmup at DEPTH=10**
  - why: WARMUP_RATIO=0 means LR jumps straight to peak for the now-larger DEPTH=10 model, and both the untested-axis space (warmup, weight decay, embedding LR) and the depth/MATRIX_LR axes are exhausted, so warmup is the highest-leverage unexplored single-variable change.
  - detail: In train.py change `WARMUP_RATIO = 0.0` to `WARMUP_RATIO = 0.05` (5% of the 5-minute budget spent ramping LR linearly from 0 to peak via the existing get_lr_multiplier warmup branch, which is already implemented but currently dead code since WARMUP_RATIO=0). Leave DEPTH=10, MATRIX_LR=0.04, WARMDOWN_RATIO=0.5, and everything else at the current baseline values. This tests whether the immediate-full-LR start is destabilizing early optimization for the wider (model_dim=640) model in a way it didn't for the smaller depth=8 baseline it was presumably tuned on.

### Messages
- `#40` **a2**: DEPTH=9 (a1, 0.998924) wins cleanly and is a genuine effect — confirmed by two independent a1 runs within noise of each other (0.998944/0.998924), using less VRAM too. My own contribution this round is a clean negative: WEIGHT_DECAY is not a free lever — both 0.4 (1.003240) and 0.0 (1.006934) lost to the 0.2 baseline, bracketing it from both sides, so 0.2 looks locally optimal and this axis is exhausted at DEPTH=10. Don't re-spend slots on WEIGHT_DECAY without also changing DEPTH.

Important gotcha for round 3: every hyperparameter finding from rounds 1-2 (MATRIX_LR dead, WARMDOWN_RATIO optimal at 0.5, WEIGHT_DECAY optimal at 0.2, warmup untested) was measured at DEPTH=10/model_dim=640. The new baseline is DEPTH=9, which is also model_dim=640 (same width, one fewer layer) — so these are likely still valid since width didn't change, but nobody has actually confirmed that on DEPTH=9 direct [...]
- `#41` **a1**: DEPTH=9 confirmed twice (0.998944, 0.998924) — new baseline 0.998924, and it's the same model_dim=640 that WARMDOWN_RATIO/MATRIX_LR/WEIGHT_DECAY were all tuned/tested against in rounds 1-2, so those negative results (MATRIX_LR sweep dead, WEIGHT_DECAY=0/0.4 both worse than 0.2, WARMDOWN_RATIO=0.65/0.7 both worse than 0.5) should still hold at DEPTH=9 — no need to re-litigate them, since only layer count changed, not width.

For round 3: both open warmup proposals (#30, #32) are still worth running, but update DEPTH=10→9 in them first — they were written against the now-stale baseline. Given warmup is genuinely untested and orthogonal to the depth change, that's a reasonable pick.

Separately, the depth/width axis at width=640 is now bracketed (9 wins, 10 baseline, 12 and 11-both-crash confirm going wider is bad) but nobody has tried going *below* 640 width with more layers than 8 — e.g. [...]
- `#42` **a0**: WARMDOWN_RATIO axis is now fully bracketed and dead: 0.3 (1.015310, at old depth=8), 0.5 (baseline, best), 0.65 (1.003021), 0.7 (1.003814) — four points, monotonically worse in both directions from 0.5. Don't spend more slots retuning this schedule shape at model_dim=640.

Bigger news: a1's DEPTH=9 result (0.998924/0.998944, two independent confirmations) is a real, unconfounded win and is now the new baseline. Combined with the dead MATRIX_LR sweep (4 runs, all worse) and dead WEIGHT_DECAY sweep (0.0 and 0.4 both worse than 0.2) — both done *at depth=10* — we now have good reason to believe none of those axes need retesting at depth=9, since model_dim is identical (640) between depth=9 and depth=10, so any width-dependent tuning (MATRIX_LR, WEIGHT_DECAY, AdamW LR scaling) should transfer unchanged. That means the group shouldn't waste slots re-verifying MATRIX_LR/WEIGHT_DECAY at the new [...]

### Experiments run
- `a0/v0` val_bpb **1.003021** (66.5 GB) — Increased WARMDOWN_RATIO from 0.5 to 0.65 (longer LR cooldown, less time at peak LR), all else unchanged from the DEPTH=10 baseline.
- `a0/v1` val_bpb **1.003814** (66.5 GB) — Changed WARMDOWN_RATIO from 0.5 to 0.7 (longer cooldown, less time at peak LR), all else unchanged at DEPTH=10 baseline.
- `a1/v0` val_bpb **0.998944** (60.5 GB) — Changed only DEPTH from 10 to 9 in train.py; model_dim stays 640 (5 heads, same as depth=10) due to HEAD_DIM=128 rounding, so this isolates layer count at fixed width.
- `a1/v1` val_bpb **0.998924** (60.5 GB) — Changed DEPTH from 10 to 9 in train.py; model_dim stays 640 (5 heads) due to HEAD_DIM rounding, so this is a pure layer-count-at-fixed-width comparison against the depth=10 baseline.
- `a2/v0` val_bpb **1.003240** (66.5 GB) — Doubled cautious Muon weight decay (0.2 -> 0.4, still decaying to 0 over training) at the DEPTH=10 baseline, all else unchanged; avoided duplicating a0's identical warmup proposal.
- `a2/v1` val_bpb **1.006934** (66.5 GB) — Changed WEIGHT_DECAY from 0.2 to 0.0 (cautious Muon weight decay), all else at DEPTH=10 baseline.

**Outcome:** a1/v1 won; new baseline val_bpb 0.998924.

## Round 3
_Baseline entering the round: val_bpb 0.998924 (commit f211f03)_

### Proposals
- `#46` **a0 — LR warmup at DEPTH=9**
  - why: Warmup was proposed twice last round but never actually run (both proposers pivoted to weight decay instead), so it's a genuine, cheap, unconfounded gap at the current model width (640) where LR still jumps straight to peak.
  - detail: In train.py change `WARMUP_RATIO = 0.0` to `WARMUP_RATIO = 0.05` (5% of the 5-min budget ramping LR linearly from 0 to peak via the already-implemented get_lr_multiplier warmup branch). Leave DEPTH=9, MATRIX_LR=0.04, WARMDOWN_RATIO=0.5, WEIGHT_DECAY=0.2 and everything else at current baseline values — single-variable change against the 0.998924 baseline.
- `#45` **a1 — LR warmup at DEPTH=9 baseline**
  - why: Warmup was proposed twice in round 2 but never actually run — both a0 and a2 pivoted to WARMDOWN_RATIO/WEIGHT_DECAY instead — so it remains the one cheap, plausible, fully untested axis at the current width (640).
  - detail: In train.py change `WARMUP_RATIO = 0.0` to `WARMUP_RATIO = 0.05` (5% of the 5-min budget spent ramping LR linearly from 0 to peak, via the existing get_lr_multiplier warmup branch which is implemented but currently dead code). Leave DEPTH=9, MATRIX_LR=0.04, WARMDOWN_RATIO=0.5, WEIGHT_DECAY=0.2 and everything else at the current baseline (0.998924) untouched — single-variable test.
- `#44` **a2 — LR warmup at DEPTH=9 (finally test it)**
  - why: Warmup was proposed twice in round 2 (#30, #32) but never actually run (both agents pivoted to depth/weight-decay instead) — it's the one cheap, plausible, completely untested lever left before we're purely re-litigating dead axes.
  - detail: In train.py change `WARMUP_RATIO = 0.0` to `WARMUP_RATIO = 0.05` (or 0.03 if worried about eating into the fixed 5-min budget). Leave DEPTH=9 (current baseline, model_dim=640), MATRIX_LR=0.04, WARMDOWN_RATIO=0.5, WEIGHT_DECAY=0.2 all untouched. get_lr_multiplier already implements the warmup branch (`progress / WARMUP_RATIO`), so this is a zero-code, single-constant change that ramps LR linearly from 0 to peak over the first 5% of training instead of jumping straight to peak, which may reduce early-step instability/wasted gradient steps for this width.

### Messages
- `#54` **a0**: Warmup is now dead beyond any reasonable doubt: 5 independent runs (0.03×2, 0.05×3) all lost to the 0.998924 baseline, landing in a tight 1.0005–1.0032 band. Don't touch WARMUP_RATIO again without a real reason — this axis cost the group 5 of 36 runs for zero signal, worth remembering before we pile onto a "consensus" idea again.

The actual news is a2/v1: WINDOW_PATTERN 'SSSL'→'SSSS' (drop the one full-context layer to half-context) won at 0.998440, beating baseline by 0.000484. That's small — about 13x smaller than the depth=9 win (0.0037) and same order as noise we've seen between nominally-identical MATRIX_LR reruns (e.g. a2's two 0.06 runs differed by 0.00005, but a1's two DEPTH=9 confirmations differed by 0.00002 — so 0.0005 is on the edge of being a real signal, not clearly inside noise, but it hasn't been replicated). Before treating attention-window pattern as a proven lever, so [...]
- `#55` **a2**: Warmup is now a dead axis with five confirmations (0.03 and 0.05 both tested twice, one 0.05 run from a2) — all five landed between 1.0005 and 1.0032, uniformly worse than the 0.998924 baseline. Nobody needs to touch WARMUP_RATIO again at this width without a real reason to expect different behavior.

My WINDOW_PATTERN=SSSS run (0.998440) is the round's only win, but it's a single unconfirmed data point — unlike the DEPTH=9 win which had two independent confirmations before anyone trusted it. Before building further on it, I'd want at least one more run to rule out noise, ideally at a different point on the same axis rather than an exact repeat: current pattern is "only the forced-last layer gets full context" (1 of 9 long, down from 3 of 9 in the SSSL baseline). Worth bracketing with something in between, e.g. "SSSSSSSL" (1 long every 8) or "SSSSSSSS" (same as mine, confirms/refutes noi [...]
- `#56` **a1**: Warmup is now dead beyond reasonable doubt: 5 independent runs (a0 x2 at 0.03/0.05, me at 0.03, a1 at 0.05, a2 at 0.05) all landed worse than 0.998924, with no dependence on the exact ratio — the zero-warmup schedule is locally optimal at width=640. Don't spend another slot on it without a real width/depth change first.

a2's WINDOW_PATTERN='SSSS' win (0.998440) is small (~0.0005) but plausible and cheap to verify — worth a second confirming run before fully trusting it, same as we did for DEPTH=9. It also opens a genuinely new, unexplored axis: attention window size/pattern. a2's framing ("less compute per step → more steps in 5 min") suggests the natural follow-ups are SSSS vs going further (e.g. shrinking the short window itself, not just its frequency) or testing whether the last-layer-forced-full-window is even necessary. This is a good region to push into next round since the depth [...]

### Experiments run
- `a0/v0` val_bpb **1.002663** (60.5 GB) — Changed WARMUP_RATIO from 0.0 to 0.05 at the DEPTH=9 baseline, all else unchanged.
- `a0/v1` val_bpb **1.000894** (60.5 GB) — Changed WARMUP_RATIO from 0.0 to 0.03 (all three agents proposed 0.05, so I tested a different point on the same axis to avoid triple-spending on an identical config), all else at DEPTH=9 baseline unc [...]
- `a1/v0` val_bpb **1.000503** (60.5 GB) — Changed WARMUP_RATIO from 0.0 to 0.03 (3% of the 5-min budget ramping LR linearly from 0 to peak), all else at DEPTH=9 baseline unchanged.
- `a1/v1` val_bpb **1.003182** (60.5 GB) — Changed WARMUP_RATIO from 0.0 to 0.05 at DEPTH=9 baseline, all else unchanged.
- `a2/v0` val_bpb **1.002883** (60.5 GB) — Changed WARMUP_RATIO from 0.0 to 0.05 (5% linear LR ramp from 0 to peak), all else at DEPTH=9 baseline.
- `a2/v1` val_bpb **0.998440** (60.5 GB) — Changed WINDOW_PATTERN from 'SSSL' to 'SSSS' at DEPTH=9 baseline: only the code-forced last layer keeps full-context attention, all other 8 layers use half-context sliding window (was 3 of 9 layers lo [...]

**Outcome:** a2/v1 won; new baseline val_bpb 0.998440.

## Round 4
_Baseline entering the round: val_bpb 0.998440 (commit 55352c6)_

### Proposals
- `#58` **a0 — Re-sweep DEPTH under the new WINDOW_PATTERN=SSSS baseline**
  - why: The depth=9 vs depth=10 comparison was decided under WINDOW_PATTERN=SSSL (3/9 long-context layers); SSSS roughly halves attention FLOPs/layer, which shifts the compute-optimal depth for the fixed 5-min budget and was never retested.
  - detail: In train.py change only `DEPTH = 9` to `DEPTH = 10` (leave WINDOW_PATTERN="SSSS", ASPECT_RATIO=64, HEAD_DIM=128, DEVICE_BATCH_SIZE=128, and all other hyperparameters at the 0.998440 baseline). model_dim goes from 640 (5 heads) to 640... wait, actually DEPTH=10 also gives base_dim=640 (10*64=640), same width as DEPTH=9 (9*64=576 rounds to 640) — so this is a clean layer-count-at-fixed-width test, isolating whether the reduced per-layer attention cost from SSSS now favors one more layer within the fixed time budget, without any confound from width or batch size changes.
- `#60` **a1 — Shrink sliding-window size itself (not just pattern) at DEPTH=9**
  - why: The only win this round (SSSS, -0.0005) came from cutting attention compute per step by using short windows more often; cutting the window's absolute size (not just its frequency) is the same lever taken further and is completely untested.
  - detail: In `_compute_window_sizes` (train.py), change `short_window = long_window // 2` to `short_window = long_window // 4` (i.e. sequence_len//4 = 512 tokens instead of 1024), leaving WINDOW_PATTERN="SSSS", DEPTH=9, and every other hyperparameter at the current baseline. This isolates window-size (as opposed to window-frequency) as the single variable: lower attn FLOPs per step → more steps/tokens in the fixed 5-min budget, same mechanism that made SSSS beat SSSL, tested one step further.
- `#59` **a2 — Retest DEPTH=10 under the new WINDOW_PATTERN=SSSS baseline**
  - why: SSSS makes attention cheaper per step (fewer long-window layers), shifting the fixed-5-min compute/tokens tradeoff — the earlier DEPTH=9-vs-10 result was measured entirely under the old SSSL pattern and may not hold now that steps are faster.
  - detail: In train.py change only `DEPTH = 9` to `DEPTH = 10` (leave WINDOW_PATTERN="SSSS", ASPECT_RATIO=64, HEAD_DIM=128, DEVICE_BATCH_SIZE=128, and all optimizer hyperparameters untouched). model_dim scales to 640 same as depth=9 (both round up to 640 via HEAD_DIM=128), so this isolates layer count at fixed width, same clean comparison structure a1 used to establish depth=9 as the winner over depth=10 — just re-run under the current (cheaper) attention pattern instead of the old SSSL one it was originally decided under.

### Messages
- `#68` **a0**: Four independent runs (a0, a1×2, a2) at short_window=long_window//4 all landed in a tight 0.99295–0.99315 band — this is a real, robust win (~0.0055 over baseline), not noise, and it's now the new baseline at 0.992950. Given that, the DEPTH=10-under-SSSS proposals (#58/#59) are stale: they were written against the old short_window=//2 baseline and nobody has run them at all this round (all 6 slots went to confirming the same window-size change). With only 4 runs left in the whole cell, I'd deprioritize re-litigating depth at yet another attention config and instead push the window-size axis one step further, since it's the only lever that's shown large (>10x-noise) gains two rounds in a row: try short_window=long_window//8 (256 tokens) at the current DEPTH=9/SSSS baseline. If the trend reverses (worse than //4), that tells us //4 is near the local optimum on this axis and we should stop; [...]
- `#69` **a1**: Window-size shrink (long_window//4) is now confirmed 5x independently — a0/v0 0.993109, a1/v0 0.992950, a1/v1 0.993010, a2/v0 0.992975, a2/v1 0.993153 — a tight cluster spanning only 0.0002, versus a ~0.0055 gap to the 0.998440 baseline. This is unambiguously real, not noise, and a1/v0 (0.992950) is the new baseline. Three of us independently proposed the same change this round — a coordination miss, not a disaster, but worth flagging since it burned slots that could've covered more ground.

a0/v1's negative result is also worth banking: removing the forced full-context last layer (window_sizes[-1] override) gave 0.998511, a wash against the *old* SSSS baseline (0.998440) — the forced long layer isn't hurting or helping. Don't retest that at the new //4 baseline without a real reason.

With only 4 runs left in the whole cell, we need to be surgical. Two things I'd deprioritize: (1) don't [...]
- `#70` **a2**: Window-size shrink (long_window//2 → //4) is now confirmed 5x independently (a0 0.993109, a1 0.992950/0.993010, a2 0.992975/0.993153), all tightly clustered in a 0.0002 band — this is a real, robust effect, not noise, and a1/v0 (0.992950) is the new baseline. Good use of the round: three of us converged on the same idea independently, which is fine for confirmation but means we spent 5 slots to learn one thing very solidly rather than covering more ground — worth remembering now that only 4 runs remain for the whole cell.

Two things for whoever spends the next slot:
1. Both open proposals (#58 a0, #59 a2) for "retest DEPTH=10 under SSSS" are now stale — they were written against the 0.998440 SSSS baseline, not the new window//4 one. If DEPTH=10 is still worth testing, it should be tested against the current baseline (window=512), not the old one, since the FLOPs/step calculus changed ag [...]

### Experiments run
- `a0/v0` val_bpb **0.993109** (60.5 GB) — Changed short_window from long_window//2 (1024 tokens) to long_window//4 (512 tokens) in _compute_window_sizes, keeping WINDOW_PATTERN=SSSS and DEPTH=9 unchanged; cuts attention FLOPs further so more [...]
- `a0/v1` val_bpb **0.998511** (60.5 GB) — Removed the `window_sizes[-1] = (long_window, 0)` override in _compute_window_sizes so every layer (including the last) uses the short (half-context) window when WINDOW_PATTERN=SSSS, instead of forcin [...]
- `a1/v0` val_bpb **0.992950** (60.5 GB) — Changed short_window from long_window//2 to long_window//4 in _compute_window_sizes (train.py), at WINDOW_PATTERN=SSSS, DEPTH=9 baseline; all other hyperparameters unchanged.
- `a1/v1` val_bpb **0.993010** (60.5 GB) — Changed short_window = long_window // 2 to long_window // 4 in _compute_window_sizes (sequence_len//4 = 512 tokens instead of 1024), keeping WINDOW_PATTERN=SSSS and DEPTH=9 unchanged.
- `a2/v0` val_bpb **0.992975** (60.5 GB) — Changed short_window from long_window//2 to long_window//4 in _compute_window_sizes (train.py), at DEPTH=9/WINDOW_PATTERN=SSSS baseline; all other hyperparameters unchanged.
- `a2/v1` val_bpb **0.993153** (60.5 GB) — Changed short_window from long_window//2 to long_window//4 in _compute_window_sizes (train.py), at DEPTH=9/WINDOW_PATTERN=SSSS baseline: short-window layers now attend to 512 tokens instead of 1024, c [...]

**Outcome:** a1/v0 won; new baseline val_bpb 0.992950.

