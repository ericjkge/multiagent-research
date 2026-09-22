# Shared research log

Baseline at start of the cell: **val_bpb 0.997333**

## Round 0
_Baseline entering the round: val_bpb 0.997333 (commit 228791f)_

### Proposals
- `#7` **a0 — Halve total batch size (2^19 → 2^18)**
  - why: A 5-minute run at 524K tokens/step gets only a few hundred Muon steps, so it is step-limited rather than data-limited; halving the batch roughly doubles optimizer steps for the same tokens and also drops grad-accum to 1.
  - detail: Single-constant change in the hyperparameter block of `train.py`:

- `TOTAL_BATCH_SIZE = 2**19` → `TOTAL_BATCH_SIZE = 2**18` (~262K tokens/step).
- Leave `DEVICE_BATCH_SIZE = 128` as is. With `MAX_SEQ_LEN = 2048`, `tokens_per_fwdbwd = 128*2048 = 262144`, so `grad_accum_steps` goes 2 → 1. The `assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0` still holds. Peak VRAM is unchanged (device batch is unchanged), and the per-step accumulation loop overhead disappears.
- Leave every LR untouched (`EMBEDDING_LR 0.6`, `UNEMBEDDING_LR 0.004`, `MATRIX_LR 0.04`, `SCALAR_LR 0.5`) and the schedule untouched. This keeps the experiment a clean one-variable test: same tokens, ~2x the updates.

One caveat worth flagging: `get_muon_momentum` ramps over a fixed `step/300`, so doubling the step count implicitly shortens the momentum warmup as a fraction of training. I would *not* change it in this run — it is a second variable — but if the run wins, the obvious follow-up is scaling that 300 with the expected step count.

Interpretation plan:
- If `val_bpb` improves, the next probe is `2**17` with `DEVICE_BATCH_SIZE = 64` (grad_accum stays 1), to find where the step-count gain turns over into gradient noi [...]
- `#5` **a1 — Halve the batch size (2^19 → 2^18 tokens/step)**
  - why: At a 5-minute budget the run only gets ~800 optimizer steps at 524K tokens/step, far above the critical batch size for a ~500-dim model, so the same token budget spent in 2× as many steps should convert into strictly more optimization progress.
  - detail: One-constant change: `TOTAL_BATCH_SIZE = 2**19` → `2**18` in the hyperparameter block. `DEVICE_BATCH_SIZE` stays 128, so `grad_accum_steps = 262144 / (128*2048) = 1` — the gradient-accumulation loop collapses to a single micro-step, which also removes a little per-step overhead (no accumulation of a second fwd/bwd before the optimizer, same tokens/sec otherwise). Nothing else changes: LR schedule is time-based (`progress = total_training_time / TIME_BUDGET`), so warmdown still lands correctly; only `get_muon_momentum(step)`'s 300-step ramp now completes in half the wall-clock, which is benign. Muon's update is norm-normalized, so its effective step size is largely batch-size-independent — I deliberately do not retune LRs in this run, so the measurement isolates the batch-size effect. Expect ~1600 steps instead of ~800 at roughly the same total tokens. If it wins, the natural follow-ups are 2^17 (needs DEVICE_BATCH_SIZE 64) and a small MATRIX_LR/EMBEDDING_LR retune on top.
- `#4` **a2 — Halve the optimizer batch size (2^19 → 2^18 tokens)**
  - why: At ~400M tokens in 5 minutes, a 524K-token batch buys only ~700-800 optimizer steps — almost certainly far above the critical batch size for this model, so halving it doubles the number of updates at identical token throughput and should cut val_bpb for free.
  - detail: One-line change in the hyperparameter block: `TOTAL_BATCH_SIZE = 2**19` → `2**18`. Leave `DEVICE_BATCH_SIZE = 128` (128 × 2048 = 262144 = 2^18), which makes `grad_accum_steps == 1` and also removes the gradient-accumulation loop overhead and the extra `zero_grad`/accumulation traffic, so steady-state tok/s should be equal or slightly better. No other code changes; the schedules are all time-based (`progress = total_training_time / TIME_BUDGET`) so they adapt automatically, and `get_muon_momentum(step)` warms up over 300 steps which is still a small fraction of the (now ~1500) total steps. Learning rates stay as-is on the first attempt: Muon's update is norm-normalized (orthogonalized + NorMuon rescaled), so its per-step size is largely gradient-magnitude independent, and the AdamW groups are also scale-invariant; if the run destabilizes or the loss curve is visibly noisier late, the cheap follow-up is EMBEDDING_LR/MATRIX_LR × ~0.7 (sqrt scaling). Report steps, tok/s, and val_bpb so the group can see whether the step-count/noise trade has flipped sign; if 2^18 wins, 2^17 (DEVICE_BATCH_SIZE 64) is the obvious next probe and would also cut VRAM.
- `#6` **a3 — U-net skip connections between encoder and decoder halves**
  - why: The net already injects the embedding into every layer (`x0_lambdas`) but has no path from mid-network representations to the second half; U-net skips are one of the most reliably positive, nearly free additions in the modded-nanogpt lineage, and here they're ~8 lines with a zero-init (identity-at-s [...]
  - detail: In `GPT.__init__`, add alongside the existing scalars:
`self.skip_lambdas = nn.Parameter(torch.zeros(config.n_layer // 2))`.
In `init_weights`, `self.skip_lambdas.fill_(0.0)` (zero-init = exact identity to the current model, so the run can only diverge from baseline where it helps).

In `GPT.forward`, replace the block loop with:
```python
n = self.config.n_layer
skips = []
for i, block in enumerate(self.transformer.h):
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    if i >= n - n // 2:
        x = x + self.skip_lambdas[i - (n - n // 2)] * skips.pop()
    ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
    x = block(x, ve, cos_sin, self.window_sizes[i])
    if i < n // 2:
        skips.append(x)
```
(depth 8 → layers 4,5,6,7 receive the outputs of layers 3,2,1,0 respectively.)

In `setup_optimizer`, add `skip_params = [self.skip_lambdas]`, include it in the `assert len(list(self.parameters())) == ...` sum, and give it its own AdamW group with the same settings as the `x0_params` group (`lr=scalar_lr`, `betas=(0.96, 0.95)`, `eps=1e-10`, `wd=0.0`). Also add `self.skip_lambdas.numel()` into the `scalars` entry of `num_scaling_params` and in [...]
- `#3` **a4 — Halve the token batch (2^19 → 2^18)**
  - why: At a 5-min horizon the run gets only a few hundred optimizer steps, and 524K tokens/step is far above the critical batch size for a 512-dim/8-layer model, so halving it roughly doubles the step count for the same tokens — and it also drops grad accumulation to 1, removing a little overhead.
  - detail: One-line change in the hyperparameter block of `train.py`:

- `TOTAL_BATCH_SIZE = 2**19` → `TOTAL_BATCH_SIZE = 2**18` (262144).

Nothing else changes. `DEVICE_BATCH_SIZE = 128` and `MAX_SEQ_LEN = 2048` give `tokens_per_fwdbwd = 262144`, so `grad_accum_steps` goes 2 → 1; the assert `TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0` still holds. Peak VRAM is unchanged (the micro-batch is the same); throughput should be equal or slightly better since the accumulation loop disappears. All LRs are held fixed deliberately: this is a single-variable test, and the implied increase in LR-per-token is in the direction that usually helps at short horizons. The time-based schedules (`progress = total_training_time / TIME_BUDGET`) are batch-agnostic, so warmdown still lands correctly; only `get_muon_momentum(step)`'s 300-step ramp becomes relatively shorter in wall-clock, which is fine/desirable.

Reported `num_steps` in the summary is the direct diagnostic: if steps roughly double and `val_bpb` drops, the next obvious probe is 2^17, and if it *rises*, we have learned that the baseline batch was already at or below the critical size and the whole batch axis can be dropped for the rest of the cell.
- `#2` **a5 — Halve the optimizer batch (2^19 → 2^18 tokens)**
  - why: In a 5-minute run the model only takes ~200–250 optimizer steps at 524K tokens/step, far above the critical batch size for this horizon, so trading tokens-per-step for 2× more steps should buy real loss.
  - detail: One-line change: `TOTAL_BATCH_SIZE = 2**19` → `2**18` in the hyperparameter block. With `DEVICE_BATCH_SIZE = 128` and `MAX_SEQ_LEN = 2048`, `tokens_per_fwdbwd = 262144`, so `grad_accum_steps` drops 2 → 1 (the assert still holds); total tokens processed stays essentially the same, the number of optimizer steps roughly doubles, and the time-based LR/WD/momentum schedules need no edits since they key off `total_training_time / TIME_BUDGET`, not step count. Only `get_muon_momentum(step)`'s 300-step ramp becomes a slightly smaller fraction of training, which is fine (it just reaches 0.95 at the same wall-clock-ish point relative to a longer step count — if anything, stretch to 600 steps as a trivial follow-up, not part of this run). No VRAM change (activation memory is set by `DEVICE_BATCH_SIZE`, untouched); MFU should be unchanged or marginally better from one fewer accumulation boundary. Keep all LRs fixed for a clean single-variable read: Muon's per-update orthogonalization normalizes gradient scale, so it tolerates batch changes far better than plain Adam, and holding LRs constant means any delta is attributable to steps-vs-batch alone. If this improves, the obvious next probe is 2^ [...]

### Messages
- `#8` **a4**: a4 (#3): five of six proposals this round are the same 2^19 -> 2^18 change. a0/a1/a2/a5 have it covered, so I am NOT running it — I am taking the next point on the same axis instead: TOTAL_BATCH_SIZE = 2**17 with DEVICE_BATCH_SIZE = 64 (grad_accum still 1, VRAM roughly halves, eval batch 64 still divides EVAL_TOKENS exactly -> 160 eval steps). LRs untouched, same single-variable discipline. Between us the round then returns a slope over {2^19, 2^18, 2^17} rather than one number measured four times: if 2^18 wins and 2^17 wins more, keep descending; if 2^18 wins and 2^17 loses, we have bracketed the optimum in one round and the batch axis is closed.
- `#9` **a0**: a0 (#7): five of six of us proposed the identical 2**18 run. I am NOT running it — a fifth copy of the same measurement is a wasted slot. Taking the next point on the same curve instead: TOTAL_BATCH_SIZE = 2**17 with DEVICE_BATCH_SIZE = 64 (grad_accum stays 1, VRAM halves), all LRs and schedules untouched. Between whoever runs 2**18 and me, the round gets a 3-point curve (2**19 baseline / 2**18 / 2**17) for two slots instead of one point for five. Eval stays comparable: EVAL_TOKENS=40*524288 divides evenly by 64*2048, so evaluate_bpb sees the same 20.97M val tokens. If 2**18 wins and 2**17 wins by more, the axis has not turned over yet and someone should probe 2**16 next round; if 2**17 is worse than 2**18, we have bracketed the optimum in one round. Suggest the remaining duplicates switch to an untouched axis — HEAD_DIM 64 (8 heads instead of 4 at n_embd 512) is a one-line, zero-FLOP-ch [...]
- `#10` **a2**: a2: five of us (#7,#5,#4,#3-by-a4,#2 and my own #4) proposed the identical 2^19->2^18 batch halving. Four agents running the same one-line change is four wasted slots, so I am NOT running a fifth. Instead I take 2^18 *plus* one orthogonal change: HEAD_DIM 128 -> 64, i.e. 8 heads of 64 instead of 4 heads of 128 at the same n_embd=512. This is exactly zero delta in params, FLOPs and VRAM (n_head*head_dim = n_embd either way; attn flops term is 12*h*q*window = 12*n_embd*window regardless) - it only changes how the residual width is partitioned across attention heads and how rotary is applied. 4 heads at depth 8 is unusually few (GPT-2/modded-nanogpt sit at head_dim 64), so I expect more head diversity to buy a small real gain. Because the pure-2^18 runs are my control, this round still gives the group a clean read on head count: my val_bpb vs theirs isolates HEAD_DIM exactly. Nobody has cla [...]
- `#11` **a1**: a1: four of us (#7 a0, #4 a2, #3 a4, #2 a5) proposed the identical 2^18 change, including me (#5). Four redundant runs out of 36 is a bad trade, so I am NOT running 2^18 — someone else clearly will. I am taking the next point on the same axis instead: TOTAL_BATCH_SIZE=2^17 with DEVICE_BATCH_SIZE=64 (grad_accum stays 1). That way this round returns the *shape* of the batch axis, not one point on it: baseline 2^19, their 2^18, my 2^17. If 2^18 wins and 2^17 wins more, go smaller next round; if 2^17 is worse than 2^18, we have the turnover and the axis is closed in one round. Caveat I expect to see: at 2^17 the fixed per-step optimizer cost (Muon NS iterations over ~100M params) is amortized over half the tokens, so tok/s should drop ~10%; I will report it. Suggestion to whoever is still deciding: at least one more of you should drop 2^18 and take something orthogonal — a3's U-net skips (#6 [...]
- `#12` **a5**: a5: five of six proposals were the same batch halving, so I'm not spending a sixth run on it. I'm taking the untouched model-size axis instead, on top of the consensus batch: TOTAL_BATCH_SIZE=2**18 AND DEPTH=8->10 (dim 512->640, 5 heads), DEVICE_BATCH_SIZE 128->64 for VRAM headroom (grad_accum=2, same total batch). Rationale: depth/width is the largest untested lever and nobody else is touching it; measured against the 2**18-only arm that four agents are running, my run reads directly as 'does depth 10 pay for its ~1.6x FLOPs/token in 5 minutes?'. If I lose to plain 2**18, that is a clean negative for going wider/deeper and the group should stop considering it.
- `#13` **a4** _(re: #8)_: a4 RESULT, 2^17 (device batch 64): val_bpb 0.996530 vs 0.997333 baseline — a real but SMALL win, -0.0008. num_steps 3205, total_tokens 420.1M, steady-state MFU 33.72%, peak_vram 22.7GB.

Two things for the group:

1. Step-count calibration, which several of you asked for: ~3200 steps at 2^17 means the 2^19 baseline was getting ~800. The shared premise behind all five identical proposals ('only a few hundred optimizer steps') was correct.

2. The win is far smaller than that premise predicted, and the run log says why. Per-step dt bottoms out at ~80ms but averages ~105ms, and steady-state MFU is 33.7% against a 39.6% peak: at 131K tokens/step a meaningful slice of the time budget goes to per-step launch + optimizer overhead rather than matmuls. So this axis is a tug-of-war, not a free lunch — 4x the updates also bought 4x less data per update PLUS an overhead tax, and the two nearly cance [...]
- `#14` **a3**: a3 result (#6, U-net skips): NEGATIVE. val_bpb 1.001167 vs baseline 0.997333, +0.0038 — a regression, not a wash. 927 steps, 486M tokens, MFU 38.7%, 45.1GB, i.e. zero throughput cost; the damage is purely representational. Gates were zero-init (exact identity at step 0), so the optimizer actively moved skip_lambdas somewhere harmful.

My reading: this net already has TWO cross-layer shortcut mechanisms — per-layer x0 injection (x0_lambdas) and value embeddings on alternating layers. U-net skips are a third, and it looks like redundant capacity that only adds gradient interference. I am not going to spend another shared run rescuing this with a smaller scalar LR or fewer skips. Treat 'add another residual/shortcut path' as a closed region for this codebase.

Note that 5 of 6 of us proposed the identical 2^19->2^18 batch halving this round, so the cell likely spent 5 runs to learn one bit. [...]
- `#15` **a0** _(re: #9)_: a0 RESULT (2**17, device batch 64): val_bpb 0.997280 vs 0.997333 baseline. That is a TIE, not a win - 5e-5 on a metric we are trying to move by 1e-2.

The numbers that matter: num_steps 3153, total_tokens 413.3M, mfu 33.17%, peak_vram 22.7GB.

The baseline at 2**19 does ~790 steps over the same ~413M tokens. I quadrupled the update count at constant tokens and val_bpb did not move. So the shared premise of this round - 'a 5-minute run is step-limited, buy more steps' - is wrong. Several proposals (mine included, and a5's explicitly) guessed 200-800 steps and treated that as obviously too few; it is not too few, it is already past the point where more updates help. This regime is throughput-limited, not step-limited. I would call the batch axis closed between 2**19 and 2**17 and NOT spend further slots on 2**16 - the curve is flat, not rising, so there is no optimum to bracket.

Whoever r [...]
- `#16` **a2**: a2 RESULT (commit 14b65b9): val_bpb 0.993500 vs 0.997333 baseline, -0.00383. Config: TOTAL_BATCH_SIZE 2^18 AND HEAD_DIM 128->64 (8 heads of 64, n_embd still 512). num_steps 1795, total_tokens 470.5M, mfu 37.7%, peak VRAM 45.0GB, training_seconds 300.1. Read this against the pure-2^18 runs: their val_bpb is the control, and (mine - theirs) is a clean estimate of the HEAD_DIM effect, since head count changes nothing in params/FLOPs/VRAM. If they land near 0.9935 too, head count is a wash and should be dropped; if they land nearer 0.997, most of the win was head count and the batch halving was the wash.

IMPORTANT CORRECTION for everyone's scaling reasoning, including my own #10: the real vocab_size is 8192, not the 32768 in the GPTConfig dataclass default - the actual value comes from the tokenizer at runtime. So the model is 50.3M params total, of which only ~29M is lm_head + transformer [...]
- `#17` **a1** _(re: #11)_: a1 RESULT (2^17, device batch 64): val_bpb 0.997942 vs baseline 0.997333 — slightly WORSE with 4x the optimizer steps. Details: 3101 steps (baseline ~775), 406.5M total tokens (i.e. the SAME tokens as baseline), 1.63M tok/s, MFU 32.6%, peak VRAM 22.7GB. Throughput is not the confound — tokens matched — so this is a clean read on optimization quality alone, and it does not support the step-limited premise that five of six of us proposed. Reading for the group: the batch axis is now bracketed at 2^19 (0.997333) and 2^17 (0.997942). Whatever the 2^18 runs report, it can only be a shallow dip between two near-equal endpoints, not the large win the argument predicted; four runs at the same 2^18 point will all confirm each other and teach us one number. Two honest caveats on my own result: (1) I held all LRs fixed on purpose, so this tests small-batch-at-baseline-LR, not small-batch-properly-t [...]
- `#18` **a5**: a5 RESULT (#12): depth 10 + batch 2**18 -> val_bpb 0.988076 (baseline 0.997333, -0.0093). 1119 steps, 293.3M tokens, MFU 41.9%, peak VRAM 34.6GB, 85.9M params (dim 640, 5 heads, device batch 64 / grad_accum 2). Caveat stated plainly: this is two changes at once, so read my number against the four 2**18-only arms - if plain 2**18 lands at or below 0.988 then depth 10 bought nothing and I'd drop it; if 2**18 lands around 0.993-0.995 then depth is worth roughly half the gain and the depth axis is live. Two facts worth having regardless of how that resolves: (1) VRAM is nowhere near binding - 34.6GB of 80GB at depth 10 with device batch 64, so nobody should be sizing models around memory yet, and device batch 128 at this size should fit; (2) MFU is 41.9%, i.e. the 1.6x FLOPs/token of going 512->640 dim still left 293M tokens inside the 300s, so depth/width is cheaper here than the FLOP count [...]
- `#26` **a5** _(re: #13)_: a5 -> a4: your -0.0008 is inside the noise floor, not a real win. a0_v0, a1_v0 and a4_v0 train.py are byte-identical except one space in a comment on line 451 (I diffed them), i.e. three replicates of the same config: 0.996530 / 0.997280 / 0.997942, range 0.00141, sd 0.00071. Your run is simply the lucky one of the three. a0's reading in #15 ('a tie') is the correct one, and the replicate spread is the round's most useful by-product: nothing below ~0.0015 should be called a result from a single run.
- `#27` **a1** _(re: #13)_: a1 -> a4 (#13): your -0.0008 at 2^17 is not a real win, and I can prove it rather than argue it. Your train.py, a0's and mine are byte-identical at 2^17/device-64 except one space in a comment (md5: a0 and a1 identical; a4 differs only by 'DEVICE_BATCH_SIZE = 64 ' vs '= 64'). Three runs of the same config gave 0.996530 / 0.997280 / 0.997942 — sd 0.00071, range 0.00141. Your -0.0008 is 1.1 sd, i.e. exactly the run-to-run spread; you drew the lucky sample and I drew the unlucky one from the same distribution. Not your fault — the seed is fixed at 42, so we all expected determinism; the nondeterminism enters through the time budget (we got 3205 / 3153 / 3101 steps, so different data and different schedule endpoints). Useful side effect: we accidentally bought the cell a noise floor. Full argument in my round message.
- `#28` **a4** _(re: #16)_: a4 -> a2: your inference plan does not have the control it assumes. You wrote 'the pure-2^18 runs are my control' and 'if they land near 0.9935 too, head count is a wash'. Nobody ran pure 2^18. a0/a1/a4 all ran 2^17 (device batch 64); a3 ran the unmodified 2^19 batch; a5 ran 2^18 but with depth 10. The 2^18 arm you were counting on does not exist in results.tsv.

The good news is you do not need it, and the group should NOT spend a slot manufacturing it. The batch axis turned out flat: baseline 2^19 = 0.997333 and the three 2^17 replicates are 0.996530 / 0.997280 / 0.997942, i.e. the baseline sits inside the replicate band. A pure 2^18 run is bracketed by two points that are already equal, so it is ~0.9973 with high confidence and a run spent confirming that buys one number we can already read off the curve.

Substituting that inferred control: your -0.0038 is mostly HEAD_DIM, and a5's - [...]
- `#29` **a0** _(re: #13)_: a0 -> a4 (#13): I have to push back on 'a real but small win (-0.0008)' and on 'the step-limited premise was correct'. Your train.py and mine are byte-identical apart from one space in a comment on line 451 — I diffed them (rounds/r00/a4_v0/train.py vs a0_v0/train.py). a1's is identical to mine too, md5 5c78ace1554a60c9530207e8f847ff8b for both. Three replicates of one config: 0.996530 / 0.997280 / 0.997942, std 0.000706. Your -0.0008 is 1.1 sigma of the very distribution your own run is a sample from; mine was +0.0001 and a1's was +0.0006 on the same code. The three of us should report this as one number, 0.99725 +/- 0.0007, not as three findings pointing three directions. The step counts differ (3205 / 3153 / 3101) because the schedule keys off wall-clock, so each replicate sees different data and lands warmdown differently — that is where the spread comes from, seed 42 notwithstanding [...]
- `#30` **a5**: a5. The round accidentally produced the most useful number in the cell: a noise floor. a0_v0, a1_v0 and a4_v0 train.py are byte-identical except one space in a comment on line 451 — I diffed them. Three replicates of one config gave 0.996530 / 0.997280 / 0.997942: range 0.00141, sd 0.00071. Seed is fixed at 42, so the variance comes from the time-based stop (3205/3153/3101 steps) changing both data seen and where warmdown lands. Apply this: (1) nothing under ~0.0015 from a single run is a result — a4's -0.0008 (#13) is noise, a0's "tie" (#15) is right; (2) the winner of a 6-run round is biased lucky by ~1σ, so expect my 0.988076 to be ~0.0007 optimistic; (3) a2's -0.0038 (~5σ) and my -0.0093 (~13σ) are both real.

Second: **nobody ran pure 2^18.** Four of us said we would, then all four peeled off. The control a2 and I said to compare against does not exist — do NOT spend a slot making i [...]
- `#31` **a2**: **We accidentally ran a triplicate. It gives us a noise floor — read it before trusting any small number from this round.**

`md5sum`: a0's and a1's `train.py` are **byte-identical** (5c78ace1); a4's differs by one space in a comment. Same seed, same config. Scores:

- a4 `8313610`: 0.996530 (3205 steps)
- a0 `ae7d02d`: 0.997280 (3153 steps)
- a1 `4ad54fc`: 0.997942 (3101 steps)

Range 1.4e-3, σ ≈ **7e-4**. Wall-clock jitter moves the step count (3101–3205) and kernels aren't bitwise deterministic, so identical code lands anywhere in a ~1.5e-3 window. Consequences:

1. **a4 (#13): your −0.0008 is not "a real but small win", it is one sample inside your own group's spread.** a0's tie reading (#15) is correct. Don't build on it.
2. **Anything under ~2e-3 from n=1 is unpublishable here.** With 30 runs left, only chase priors ≥3e-3.
3. **a3 (#14): your +0.0038 is ~5σ — that regression is rea [...]
- `#32` **a1**: **We accidentally bought a noise floor. Use it before reading any other number.**

a0, a1 and a4 ran *byte-identical* train.py (md5: a0==a1; a4 differs by one space in a comment). Same config, three results:

| | val_bpb | steps | tokens |
|---|---|---|---|
| a4 | 0.996530 | 3205 | 420.1M |
| a0 | 0.997280 | 3153 | 413.3M |
| a1 | 0.997942 | 3101 | 406.5M |

**mean 0.997251, sd 0.00071, range 0.00141.** Seed is fixed at 42, so we expected determinism; it enters through the *time* budget — throughput jitter changes step count, hence data seen and where warmdown lands. **Anything under ~0.0015 is not a result.**

Re-reading the round through it:

- **Batch axis: closed, n=3.** 2^17 mean vs baseline = -0.00008 (0.1 sd) across a 4× range. a4, your -0.0008 is 1.1 sd — you drew the lucky sample, I drew the unlucky one (#27). Nobody run 2^16, and **nobody run the missing pure-2^18 control**: th [...]
- `#33` **a4**: **The round's most valuable result is an accident: we ran the same config three times and got three different answers.**

a0's and a1's `train.py` are **byte-identical** (md5 `5c78ace1…`); mine differs from a0's by one space in a comment. Three identical runs, same seed:

| | val_bpb | tokens | steps |
|---|---|---|---|
| a4 | 0.996530 | 420.1M | 3205 |
| a0 | 0.997280 | 413.3M | 3153 |
| a1 | 0.997942 | 406.5M | 3101 |

**sd ≈ 0.0007, range 0.0014.** A measured noise floor, not a guessed one. Three consequences:

1. **I retract my own claim.** I called -0.0008 "a real but small win". It is one draw from that band; a0's "tie" was correct. Nothing under ~0.002 is a result.
2. **Batch axis dead.** Baseline 2^19 (0.997333) sits *inside* the replicate band. Don't probe 2^16 — and **don't spend a slot on pure 2^18 either**: it is bracketed by two equal points, so it is ~0.9973 for free. a2 an [...]
- `#34` **a3**: a3. Three things before round 1 picks.

**1. We accidentally measured the noise floor, and it's large.** a0/a1/a4 didn't run three things — their `train.py` are byte-identical bar one space in a comment. Three replicates of 2^17: 0.997280 / 0.997942 / 0.996530. Mean 0.997251, **sd 7.1e-4, range 1.4e-3** (they differ because the loop is wall-clock-bound: 3153/3101/3205 steps). So:
- a4's "real but small win" of -0.0008 (#13) is 1.1σ. Not a result — a4's arm is the middle of the three, the other two bracket it both ways.
- 2^17 mean vs baseline: **-8e-5**. Batch axis is flat *with an error bar*. Closed. Nobody run 2^16.
- New bar: under ~1.5e-3 is not a win. Against it: a2 -5.4σ, a5 -13.1σ, my U-net +5.4σ. All three real.

**2. The pure-2^18 control doesn't exist.** Five proposed it, all five fled. a2 (#16) and a5 (#18) both wrote their interpretation around "the four 2^18-only arms" — the [...]
- `#35` **a0**: 

### Experiments run
- `a0/v0` val_bpb **0.997280** (22.2 GB) — TOTAL_BATCH_SIZE 2**19 -> 2**17 and DEVICE_BATCH_SIZE 128 -> 64 (grad_accum stays 1); all LRs and schedules untouched.
- `a1/v0` val_bpb **0.997942** (22.2 GB) — TOTAL_BATCH_SIZE 2**19 -> 2**17 and DEVICE_BATCH_SIZE 128 -> 64 (grad_accum stays 1); all LRs and schedules untouched.
- `a2/v0` val_bpb **0.993500** (44.0 GB) — TOTAL_BATCH_SIZE 2**19 -> 2**18 (grad_accum 2 -> 1) plus HEAD_DIM 128 -> 64, giving 8 heads of 64 at unchanged n_embd=512 (identical params, FLOPs and VRAM).
- `a3/v0` val_bpb **1.001167** (44.0 GB) — Added 4 zero-init learnable-gated skips (layers 4,5,6,7 each add the output of layer 3,2,1,0) plus a skip_lambdas parameter trained in the x0 scalar AdamW group; 13 lines, no other change.
- `a4/v0` val_bpb **0.996530** (22.2 GB) — TOTAL_BATCH_SIZE 2**19 -> 2**17 and DEVICE_BATCH_SIZE 128 -> 64 (grad_accum stays 1); no other changes.
- `a5/v0` val_bpb **0.988076** (33.8 GB) — DEPTH 8->10 (n_embd 512->640, 5 heads), TOTAL_BATCH_SIZE 2**19->2**18, DEVICE_BATCH_SIZE 128->64 (grad_accum 2) for VRAM headroom.

**Outcome:** a5/v0 won; new baseline val_bpb 0.988076.

## Round 1
_Baseline entering the round: val_bpb 0.988076 (commit c4400da)_

### Proposals
- `#38` **a0 — HEAD_DIM 128 → 64 on the depth-10 baseline**
  - why: Head count is the one measured ~5σ win from round 0 (a2: −0.0038) that the new depth-10 baseline does not yet contain, and it costs exactly zero params, FLOPs or VRAM — the highest expected gain per run on the table.
  - detail: One-line change in the hyperparameter block of `train.py`:

- `HEAD_DIM = 128` → `HEAD_DIM = 64`.

Nothing else changes. Mechanism: `build_model_config(DEPTH)` computes `base_dim = DEPTH * ASPECT_RATIO = 10 * 64 = 640`, then `model_dim = ceil(640/HEAD_DIM)*HEAD_DIM`. At HEAD_DIM 128 that is 640 with `num_heads = 5`; at HEAD_DIM 64 it is still exactly 640 with `num_heads = 10`. So `n_embd` is unchanged at 640 — identical parameter count (c_q/c_k/c_v/c_proj are all n_embd→n_embd either way), identical FLOPs (`estimate_flops`'s attention term is `12*h*q*window = 12*n_embd*window` regardless of the split), identical activation memory, identical `kv_dim = n_kv_head*head_dim = 640` so value embeddings and `ve_gate` (which outputs `n_kv_head` channels, 5→10) are structurally unchanged. The only real differences: the residual width is partitioned into 10 subspaces instead of 5, and rotary is precomputed at head_dim 64 instead of 128 (the standard GPT-2/modded-nanogpt setting — 5 heads at depth 10 is unusually few).

Keep `DEPTH = 10`, `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64`, and every LR untouched, so this reads as a clean single-variable delta against the 0.988076 baseline. [...]
- `#42` **a1 — Depth 12 + HEAD_DIM 64 (dim 768, 12 heads)**
  - why: The only two real wins in round 0 were on orthogonal axes — depth 8→10 (−0.0093) and HEAD_DIM 128→64 (−0.0038) — so stacking them and taking the next step up the still-rising depth curve is the highest-expected-gain single run available.
  - detail: Three constants in the hyperparameter block of `train.py`, nothing else:

- `HEAD_DIM = 128` → `HEAD_DIM = 64`
- `DEPTH = 10` → `DEPTH = 12`
- `DEVICE_BATCH_SIZE = 64` (unchanged), `TOTAL_BATCH_SIZE = 2**18` (unchanged) → `grad_accum_steps = 262144 / (64*2048) = 2`.

Mechanism / arithmetic. `build_model_config` computes `base_dim = DEPTH * ASPECT_RATIO = 12*64 = 768`, rounds up to a multiple of `HEAD_DIM` (768 already is), so `n_embd = 768` and `num_heads = 768/64 = 12` (`n_kv_head` follows). HEAD_DIM 64 is exactly free: `n_head * head_dim == n_embd` either way, so params, FLOPs/token and VRAM are identical to HEAD_DIM 128 at the same dim — it only repartitions the residual across more heads and applies rotary at a shorter period. a2 measured −0.0038 for that repartition at depth 8 (≈5σ against the measured 7e-4 noise floor), and 12 heads of 64 at dim 768 is the standard GPT-2/modded-nanogpt shape.

Depth 12 raises FLOPs/token ~1.45× over the depth-10 baseline (dim 640→768, 10→12 layers), so I expect ~700–800 steps and ~200M tokens instead of a5's 1119/293M. That is the bet: a5's run showed MFU *rising* with size (38.7% at depth 8 → 41.9% at depth 10), i.e. this GPU is under-utilis [...]
- `#37` **a2 — HEAD_DIM 64 on the depth-10 baseline (10 heads instead of 5)**
  - why: My round-0 run measured HEAD_DIM 128→64 as a real ~5σ win (-0.0038) at depth 8, and the new depth-10 baseline inherited none of it — it still runs only 5 heads of 128 at dim 640, an even more head-starved ratio, so the same free change should compound with a5's depth win.
  - detail: One-constant change in the hyperparameter block of `train.py`:

- `HEAD_DIM = 128` → `HEAD_DIM = 64`.

Nothing else. Mechanism via `build_model_config(DEPTH)` with `DEPTH = 10`:
- `base_dim = 10 * ASPECT_RATIO(64) = 640`; `model_dim = ceil(640/64)*64 = 640` — **unchanged** (640 is divisible by both 64 and 128).
- `num_heads = 640 // 64 = 10` instead of `640 // 128 = 5`. `n_kv_head` follows, so it stays full MHA.

Exactly zero delta in parameters, FLOPs and VRAM:
- `head_dim = n_embd // n_head` keeps q/k/v/proj at 640×640; value-embedding `kv_dim = n_kv_head * head_dim = 640` unchanged, so `value_embeds` are the same size.
- The attention FLOP term in `estimate_flops` is `12 * h * q * window = 12 * n_embd * window`, invariant to the partition.
- Only two things change: (a) the residual width is split across 10 attention subspaces instead of 5, and (b) rotary is precomputed over `head_dim = 64` instead of 128, so each head spans 32 frequency pairs over the same 10000 base — the standard GPT-2/modded-nanogpt configuration.

The secondary effect to watch is throughput: 10 smaller FA3 heads could be less efficient than 5 large ones. At depth 8 it was not — my round-0 run held 37.7% MFU. [...]
- `#41` **a3 — HEAD_DIM 128 → 64 at the new depth-10 baseline**
  - why: a2 measured −0.0038 (≈5σ) for 8 heads of 64 vs 4 of 128 at dim 512; the new baseline is dim 640 with only 5 heads of 128, even more head-starved, and the change is exactly zero-cost in params/FLOPs/VRAM so it should compose with a5's depth win rather than compete with it.
  - detail: One-constant change in the hyperparameter block of `train.py`:

- `HEAD_DIM = 128` → `HEAD_DIM = 64` (the line `HEAD_DIM = 128          # target head dimension for attention`).

Nothing else changes. Mechanically, `build_model_config(DEPTH)` computes `base_dim = DEPTH * ASPECT_RATIO = 10 * 64 = 640`, then `model_dim = ceil(640 / HEAD_DIM) * HEAD_DIM`. With HEAD_DIM=64 that is still exactly 640 (640 is divisible by both 64 and 128), so `n_embd` is **unchanged**; only `num_heads = 640 // 64 = 10` instead of 5, and `n_kv_head = n_head = 10`.

Consequences, all checked against the code:
- Params: identical. `c_q/c_k/c_v/c_proj` are all `n_embd → n_head*head_dim = n_embd`; `kv_dim = n_kv_head * head_dim = n_embd`, so value embeddings are the same size. `ve_gate` goes `Linear(32, 5)` → `Linear(32, 10)`, i.e. +320 params across 5 gated layers — noise.
- FLOPs: identical. `estimate_flops` has `attn_flops += 12 * h * q * effective_seq` and `h*q = n_embd` either way.
- VRAM: identical (same activation tensor shapes, just reshaped).
- Rotary: `head_dim` drops 128 → 64, so `_precompute_rotary_embeddings` builds 32 frequency pairs instead of 64 over the same base=10000 — a genuinely different ( [...]
- `#40` **a4 — Depth 12 (dim 768) + fold in HEAD_DIM 64**
  - why: Depth 8→10 was the only large win in the cell (−0.0093, ~13σ) and hasn't turned over, and a2's HEAD_DIM 128→64 (−0.0038, ~5σ, zero cost in params/FLOPs/VRAM) is a confirmed win that never got merged into the new baseline — this run banks the merge and takes the next point on the live depth axis.
  - detail: Two constants in the hyperparameter block of `train.py`:

- `HEAD_DIM = 128` → `HEAD_DIM = 64`
- `DEPTH = 10` → `DEPTH = 12`

Leave `DEVICE_BATCH_SIZE = 64`, `TOTAL_BATCH_SIZE = 2**18` (grad_accum stays 2), `ASPECT_RATIO = 64`, and every LR untouched.

Mechanism / what this actually builds. `build_model_config` computes `base_dim = DEPTH * ASPECT_RATIO = 768`, then rounds up to a multiple of `HEAD_DIM`: 768 is already a multiple of 64, so `n_embd = 768` exactly and `num_heads = 12`. No rounding slack, no assert risk (`n_embd % n_head == 0` holds). So: 12 layers, dim 768, 12 heads of 64, vs the baseline's 10 layers, dim 640, 5 heads of 128.

The HEAD_DIM change is not really a second experimental variable — it is free (n_head × head_dim = n_embd either way, so params, FLOPs, activation memory and the `estimate_flops` attn term `12*h*q*window = 12*n_embd*window` are all identical), and a2 measured it at ~5σ against the round-0 noise floor. Not merging a confirmed free win costs the cell a run later. The variable under test is depth 10 → 12.

Worth flagging, not changing: `dmodel_lr_scale = (model_dim/768)**-0.5` goes 1.095 → 1.000 as dim moves 640 → 768, i.e. the AdamW groups get ~9% [...]
- `#39` **a5 — HEAD_DIM 128 → 64 at depth 10 (merge a2's win into the new baseline)**
  - why: a2 measured −0.0038 (~5σ above the 7e-4 noise floor) for head_dim 64 at depth 8, and that win was never merged — the current depth-10 baseline still runs only 5 heads of 128, so the same free-in-FLOPs/params/VRAM change should carry over.
  - detail: One-constant change in the hyperparameter block of `train.py`:

- `HEAD_DIM = 128` → `HEAD_DIM = 64`.

Nothing else changes. Mechanism, verified against `build_model_config`: `base_dim = DEPTH * ASPECT_RATIO = 10 * 64 = 640`; `model_dim = ceil(640/64)*64 = 640` (unchanged, same as with HEAD_DIM 128); `num_heads = 640/64 = 10` (was 5). So `n_embd` stays 640 and every weight matrix keeps its exact shape — `c_q/c_k/c_v/c_proj` are all `n_embd → n_head*head_dim = n_embd`. Parameter count, optimizer groups (Muon groups are keyed by `p.shape`, unchanged), and peak VRAM are identical to the current baseline. FLOPs are identical too: `estimate_flops` has `attn_flops = 12 * h * q * window`, and `h*q = n_embd` either way, so `num_flops_per_token` and reported MFU stay directly comparable. The only things that change are (a) how the 640-wide residual is partitioned across attention heads, and (b) rotary — `_precompute_rotary_embeddings` gets `head_dim=64`, so 32 frequency pairs over the same base 10000 instead of 64. QK-norm (`norm(q), norm(k)`) is applied per-head, so it now normalizes over 64 channels instead of 128; that is the standard modded-nanogpt/GPT-2 configuration, not an exotic one [...]

### Messages
- `#43` **a3**: a3: I proposed HEAD_DIM 64 at depth 10 (#41), and so did a0 (#38), a2 (#37) and a5 (#39) — four identical runs, exactly the round-0 failure mode. a1 (#42) and a4 (#40) both take depth 12 + HD64. So the round already buys {depth10+hd64} x4 and {depth12+hd64} x2 out of six slots.

**I am NOT running HEAD_DIM 64 at depth 10.** Three of you have it covered. Saying so before I launch, per the discipline point that cost us five slots last round.

I am taking the third point on the depth/size axis instead: **DEPTH = 14, HEAD_DIM = 64 (dim 896, 14 heads), DEVICE_BATCH_SIZE = 32 (grad_accum 4, same 2^18 total batch).** Rationale: depth 8->10 is the only large measured win in the cell (-0.0093, ~13sigma) and nobody knows where it turns over. With a1/a4 at 12 and me at 14, the round returns the *shape* of the size curve {10, 12, 14} at fixed HD64 rather than one more point on it. If 12 wins and 14 [...]
- `#44` **a1**: a1 SLOT CLAIM: NOT running HEAD_DIM 64 at depth 10. Four of you (#38 a0, #37 a2, #41 a3, #39 a5) proposed exactly that one-constant change, and a4 (#40) plus me (#42) proposed depth 12 + HEAD_DIM 64. That is 4 replicates of one free change and 2 of another - the round-0 mistake repeating, and we have 30 runs left.

So I am taking the third point on the depth curve: **DEPTH 14, HEAD_DIM 64, DEVICE_BATCH_SIZE 32** (dim 896, 14 heads, TOTAL_BATCH_SIZE stays 2^18 so grad_accum becomes 4). Device batch drops for VRAM: activations scale ~depth*dim, so 34.6GB at depth10/db64 would become ~68GB of 80 at db64; db32 puts it near ~35GB with room for the bigger optimizer state. 32*2048 still divides EVAL_TOKENS exactly (320 eval steps), so the metric stays comparable.

Why this and not a fifth head_dim run: depth is the ONLY large effect the cell has found (-0.0093, ~13 sigma) and nobody knows where [...]
- `#45` **a4**: a4 (#40): CLAIMING DEPTH 14, not depth 12. Arm allocation is lopsided: a0/a2/a3/a5 all proposed the plain HEAD_DIM 64 merge (four runs on a change we already measured at -0.0038), a1 (#42) and I both proposed depth 12 + HEAD_DIM 64. Four slots on the smaller known win and one axis point on the bigger unknown one is backwards. a1 has depth 12; I am taking the next point so the round returns the SHAPE of the depth curve at matched head_dim: {10 = merge arm, 12 = a1, 14 = me}.

Config: DEPTH 14, HEAD_DIM 64 -> n_embd 896 (14*64, exact, no rounding), 14 heads of 64. DEVICE_BATCH_SIZE 64 -> 32 for VRAM headroom (grad_accum 8 at TOTAL_BATCH_SIZE 2**18; EVAL_TOKENS/65536 = 320 exactly, so eval is unchanged). All LRs untouched.

Stated before the fact so I cannot rationalise afterwards. ~201M params, FLOPs/token ~2.6x the depth-10 baseline, so expect ~350-450 steps and only ~100-120M tokens. Chi [...]
- `#46` **a2**: a2: four of us (#38 a0, #37 me, #41 a3, #39 a5) proposed the IDENTICAL pure HEAD_DIM 64 at depth 10, and two (#42 a1, #40 a4) proposed depth 12 + HEAD_DIM 64. Both points are covered several times over. I am dropping my own proposal (#37) rather than be the fourth copy of it — round 0 already showed what four duplicates buy.

I am taking the THIRD point on the depth curve instead: DEPTH 14, HEAD_DIM 64 (dim 896, 14 heads of 64), DEVICE_BATCH_SIZE 64 -> 32 so grad_accum goes 2 -> 4. TOTAL_BATCH_SIZE stays 2**18, all LRs untouched. The round then returns the SHAPE of the only axis that has ever moved this metric: depth 10 (0.988076) / depth 12 (a1,a4) / depth 14 (me), all at HEAD_DIM 64. If 12 wins and 14 wins more, keep climbing next round; if 14 loses to 12 we have bracketed the optimum in one round and can stop spending slots on depth.

Two things on the record. (1) VRAM is now the bind [...]
- `#47` **a0**: a0: I proposed HEAD_DIM 64 (#38) and said I would drop it if two others claimed it. Four of us claimed it (a0 #38, a2 #37, a3 #41, a5 #39) and two more (a1 #42, a4 #40) fold it into depth 12. So the head-count arm has n=4 and needs no fifth copy, and the depth axis has n=2. I am NOT running either. Last round we spent 5 of 6 slots on one bit; not repeating that.

I am taking the WIDTH axis, which nobody has touched: ASPECT_RATIO 64 -> 80 AND HEAD_DIM 64, DEPTH stays 10. base_dim = 10*80 = 800, rounded up to a multiple of 64 -> n_embd 832, 13 heads of 64, 10 layers (~1.57x FLOPs/token over the 0.988076 baseline).

Why width rather than a fifth head run or a third depth run: a5's -0.0093 changed depth 8->10 AND width 512->640 together, so nobody knows which half of the cell's only large win was depth and which was width. a1/a4 move BOTH again (12 layers, dim 768). My arm holds depth at 10 [...]
- `#48` **a5**: a5 ROUND 1 SLOT CLAIM — I am NOT running HEAD_DIM 64 at depth 10, and I think four of us on it is round 0 repeating itself.

Tally of the proposals: a0 (#38), a2 (#37), a3 (#41) and my own (#39) are the identical one-line HEAD_DIM 128->64 at depth 10. a1 (#42) and a4 (#40) are the identical two-line depth 12 + HEAD_DIM 64. So of six slots, four buy one number and two buy a second. That is exactly the failure mode we all wrote essays about after round 0 — and with the noise floor at sd 7e-4, four replicates of HEAD_DIM 64 will also hand the cell a winner that is ~1 sigma lucky.

I am taking the one axis both clusters leave untouched: WIDTH AT FIXED DEPTH. ASPECT_RATIO 64 -> 76 and HEAD_DIM 128 -> 64, DEPTH stays 10. build_model_config: base_dim = 10*76 = 760, rounded up to a multiple of 64 -> n_embd = 768 exactly, 12 heads of 64. So 10 layers at dim 768, versus a1/a4's 12 layers at dim 76 [...]
- `#49` **a1** _(re: #44)_: a1 RESULT (#44, depth 14 + HEAD_DIM 64, dim 896, device batch 32): **val_bpb 1.033354 vs the 0.988076 baseline. +0.045, about 64 sigma. The depth axis has turned over, hard.**

Numbers: 478 steps, 125.3M tokens, 200.9M params, mfu 43.38%, peak VRAM 34.4GB, training_seconds 300.5.

The failure mode is worth being precise about, because it is NOT what I would have guessed. Throughput was fine - MFU 43.4% is the highest number in the cell (a5 got 41.9% at depth 10, a2 37.7% at depth 8). Bigger keeps getting more efficient per FLOP. The model simply ran out of data: 125M tokens against 293M at depth 10, i.e. 0.6 tokens per parameter. The last logged step reads 'loss: 2.899 | lrm: 0.00' - loss still dropping steeply while the warmdown had already finished. That is a model cut off mid-descent, not one that converged badly.

What this closes, and what it does not:

1. **Depth is bracketed: 10 - [...]
- `#50` **a3** _(re: #43)_: a3 RESULT (#43, depth 14 + HD64): **strong NEGATIVE, and it closes the size axis.** val_bpb **1.033838** vs the 0.988076 baseline — **+0.0458, ~65 sigma**. No ambiguity here.

Numbers: 476 steps, 124.8M tokens, 200.9M params (dim 896, 14 heads), **MFU 43.25%**, peak VRAM 34.4GB at device batch 32, training_seconds 300.1.

**The diagnostic everyone should take from this is the MFU.** 43.25% is the *highest* number anyone has posted in this cell, and MFU has risen monotonically with model size: 37.7% (d8, a2) -> 41.9% (d10, a5) -> 43.3% (d14, me). Bigger models use this GPU strictly better. And they still lose badly. So the size axis is not limited by throughput at all — it is limited by data:

| depth | dim | params | tokens | tok/param | val_bpb |
|---|---|---|---|---|---|
| 8 | 512 | 50.3M | 470M | 9.3 | 0.9935 (a2, +HD64) |
| 10 | 640 | 85.9M | 293M | 3.4 | 0.9881 (a5) |
| 14 | 896 | 2 [...]
- `#51` **a3** _(re: #50)_: a3 correction to #50: my shell ate three backticked names in the last paragraph. The untouched regions I meant are: (a) WARMDOWN_RATIO = 0.5 — half the budget spent decaying; (b) WINDOW_PATTERN = "SSSL" — attention context, untouched all cell; (c) the LR block, never retuned at the new size — note that dmodel_lr_scale = (model_dim/768)**-0.5 is applied ONLY to the AdamW groups; Muon MATRIX_LR = 0.04 is passed through with no dim scaling at all, so every size change we have made has implicitly shifted the Muon/Adam LR ratio. Nothing else in #50 is affected — the result stands: val_bpb 1.033838, +0.0458.
- `#52` **a4** _(re: #45)_: a4 RESULT (#45, depth 14 + head_dim 64): STRONG NEGATIVE. val_bpb 1.033314 vs 0.988076 baseline = +0.0452, about 64 sigma. 478 steps, 125.3M tokens, 200.9M params, MFU 43.39%, peak VRAM 34.4GB.

I said in #45 I thought this was more likely to lose than win and that a loss would bracket the optimum today. It lost, so here is the bracket.

My run includes HEAD_DIM 64, worth about -0.0038 by a2's measurement, so depth 14 alone cost roughly +0.049. The depth axis has turned over hard somewhere between 10 and 14.

Two facts that rule out the easy excuses:
1. It is NOT throughput. MFU rose monotonically with size: 38.7% at depth 8, 41.9% at depth 10, 43.39% here. Biggest model, best GPU utilisation. The model is starved of data, not of FLOPs - 125.3M tokens for 200.9M params.
2. It is NOT VRAM. 34.4GB of 80GB at device batch 32.

Quantitative bracket, three points made head64-consistent (addin [...]
- `#53` **a2** _(re: #46)_: a2 RESULT (depth 14, dim 896, 14 heads of 64, device batch 32 / grad_accum 4): val_bpb 1.033433 vs 0.988076 baseline. That is +0.045 — a LARGE regression, ~64 sigma against the 7e-4 noise floor. The depth axis turns over hard somewhere between 10 and 14.

Numbers: 478 steps, 125.3M tokens, mfu 43.38%, peak_vram 34.4GB, 200.9M params, training_seconds 300.5.

Three things the group should take from this.

1. **Throughput is not the confound, so the verdict is clean.** I flagged before the run that grad_accum 4 might handicap me relative to the depth-12 arms at grad_accum 2. It did not: 43.38% MFU is the HIGHEST number anyone has posted in this cell, above a5's 41.9% at depth 10 and a3's 38.7% at depth 8. Bigger matmuls keep paying for themselves on this GPU right up to dim 896. The loss is entirely representational/statistical, not a tokens-per-second artifact, so nobody needs to re-run d [...]
- `#54` **a0**: a0 RESULT (#47, width probe): STRONG NEGATIVE. ASPECT_RATIO 64->80 at DEPTH 10 (n_embd 832, 13 heads of 64, HEAD_DIM 64 folded in) -> val_bpb 1.054503 vs 0.988076 baseline, +0.0664. That is ~90 sigma on the 7e-4 noise floor and 16x larger than the HEAD_DIM-64 effect it contains, so the sign is entirely width. Clean run, no crash.

Numbers: n_embd 832, 13 heads, 130.8M params, FLOPs/token 6.72e8 (vs 4.29e8 baseline, 1.57x), 713 steps, 186.9M tokens, MFU 41.68% (baseline 41.94% - width did NOT cost utilization, my mechanical argument was right and it did not matter), peak VRAM 44.7GB, final smoothed train loss 3.043 vs baseline 2.752.

The lesson, and I think it is the most important thing this cell has learned since the noise floor:

**The size axis has turned over, and the binding quantity is tokens per parameter, not FLOPs or VRAM.**
- baseline: 85.9M params / 293M tokens = 3.4 tok/para [...]
- `#55` **a5**: a5 RESULT (#48, width at fixed depth): STRONG NEGATIVE, and it is the biggest effect anyone has measured in this cell.

Config: ASPECT_RATIO 64 -> 76 and HEAD_DIM 128 -> 64, DEPTH held at 10. That is 10 layers at dim 768 with 12 heads of 64, against the baseline's 10 layers at dim 640 with 5 heads of 128.

val_bpb 1.018287 vs baseline 0.988076 = +0.0302, about 43 sigma on the 7e-4 noise floor. 859 steps, 225.2M tokens (down from 293.3M), 114.8M params (up from 85.9M), MFU 43.84%, peak VRAM 41.3GB, final smoothed train loss 2.969 vs the baseline's 2.752.

The mechanism is NOT throughput. I predicted in #48 that width would be friendlier to the GPU than depth and it was — 41.94% -> 43.84% MFU, the highest number recorded in this cell. The bigger matmuls did exactly what I said they would, and the model still lost by 0.03. Width bought hardware efficiency and paid for it by trading 68M toke [...]
- `#63` **a2** _(re: #54)_: a2 -> a0 (#54): I have to refute 'the binding quantity is tokens per parameter', and your own run is the counterexample. I wrote the same thing in #53, so this retracts my claim too.

Your model: 10 layers, dim 832, 130.8M params, 186.9M tokens, 713 steps -> 1.054503.
The depth-14 quadruplicate: 14 layers, dim 896, 200.9M params, 125.3M tokens, 478 steps -> 1.033485.

Your model has 35% FEWER parameters, saw 49% MORE tokens, took 49% more steps, and lost by 0.021 — about 90 sigma on the r01 replicate sd of 2.4e-4. Under a tokens-per-parameter law (yours: 1.43 tok/param; theirs: 0.62) you should have won comfortably. The ordering is inverted, and not marginally.

The variable that does order all four points is aspect ratio, n_embd/n_layer:

aspect 64.0 (L10 d640, 85.9M, 3.41 t/p) -> 0.988076
aspect 64.0 (L14 d896, 200.9M, 0.62 t/p) -> 1.033485
aspect 76.8 (L10 d768, 114.8M, 1.96 t/p) -> 1 [...]
- `#64` **a3** _(re: #54)_: a3 -> a0 (#54): your own run refutes the tokens-per-parameter law you drew from it, and mine (#50) made the same mistake — I am retracting it.

Your d10/dim832 arm: 130.8M params, 186.9M tokens = 1.43 tok/param -> val_bpb 1.054503.
The d14/dim896 arm (four of us): 200.9M params, 124.8M tokens = 0.62 tok/param -> val_bpb 1.033485.

Depth 14 has 1.5x your parameters, 0.67x your tokens and 1.6x your FLOPs/token — every starvation metric is strictly worse — and it beats your arm by 0.021, about 43 pooled sigma. If tok/param were the controlling variable that ordering is impossible. val_bpb is also non-monotone in raw parameter count over the same set (130.8M scores worse than 200.9M).

What does order all five size points is aspect ratio dim/depth: at ratio 64 we get 0.9935 (d8) / 0.9881 (d10) / 1.0334 (d14), a clean U with its minimum at d10; at fixed depth 10, raising the ratio gives 64 -> [...]
- `#65` **a5** _(re: #54)_: a5 -> a0 (#54): your run and mine are the two halves of the same curve and I agree with the sign, but the rule you drew from it — 'the binding quantity is tokens per parameter' — is contradicted by the cell's own numbers, in both directions.

Your d10/dim832: 130.8M params, 186.9M tokens = 1.43 tok/param -> 1.054503.
The four d14/dim896 runs: 200.9M params, 125.3M tokens = 0.62 tok/param -> 1.03335.

The d14 arm has less than half your tokens per parameter and 70M MORE parameters, and it scored 0.021 BETTER. If tok/param ordered these results, that comparison would have to come out the other way. It also holds if you use FLOP-counted params instead of total (the estimate_flops exclusion drops wte and the value embeddings, which are 30.5% of the baseline model — 26.2M of 85.9M): 2.08 tok/eff-param for yours vs 0.88 for d14, same inversion.

What separates them is shape, not ratio: d14 sit [...]
- `#66` **a4** _(re: #54)_: a4 -> a0 (#54): your run is the best evidence in the cell AGAINST the 'tokens per parameter is the binding quantity' conclusion you drew from it, and a3 (#50) and a5 (#55) signed the same framing, so this is worth nailing down before it steers round 2.

Your width-832 arm: 130.8M params, 186.9M tokens, tok/param 1.43, val_bpb 1.054503.
The depth-14 arm (4 of us, byte-identical): 200.9M params, 125.3M tokens, tok/param 0.62, val_bpb 1.033485.

The depth-14 model is worse on BOTH of your axes - 1.5x the parameters and 0.67x the tokens, 2.3x worse tok/param - and it beat you by 0.021. Against the 2.41e-4 sd of the depth-14 quadruplicate that is ~87 sigma. If tok/param were binding, that ordering is impossible.

What the six points actually say is that growth has a per-parameter price that depends entirely on WHICH direction you grow:
  width  85.9 -> 114.8M: +0.0302 over +28.9M = +0.00104 / [...]
- `#67` **a0**: **Six runs, six regressions — and the change four of us proposed still has not been run.**

**1. We over-corrected on deconfliction.** a0/a2/a3/a5 all proposed pure HEAD_DIM 64 at depth 10, then all four dropped it to avoid duplication. `md5sum rounds/r01/*/train.py`: a1, a2, a3, a4 are **byte-identical** — four replicates of depth 14 — while the cheapest confirmed win in the cell (−0.0038) is *still unmeasured at depth 10* after 12 of 36 runs. Round 2: **exactly one agent claims pure HEAD_DIM 64, says so before launching, everyone else treats it as taken.**

**2. The noise floor is 3× tighter than we quote.** Those four identical runs: 1.033354 / 1.033433 / 1.033838 / 1.033314 — **sd 2.4e-4, range 5.2e-4**. Round 0's 7.1e-4 was n=3 (2 dof), a noisy estimate. Pooled: **sd ≈ 4.8e-4**, so the bar is ~1.5e-3 at 3σ, not 3e-3, and HEAD_DIM 64 is ~8σ. Our "only chase ≥3e-3" rule is forbidding [...]
- `#68` **a1**: **Six slots, six size probes, six negatives — and the two changes four of us proposed got zero runs.**

Four of us (me, a2, a3, a4) each wrote "I refuse to be the fourth copy of HEAD_DIM 64" and then landed on the *identical* contrarian square, depth 14. **HEAD_DIM 64 at depth 10 (proposed 4×) was never run. Depth 12 (proposed 2×) was never run.** 24 runs left; 12 of 12 spent on batch and model size, 0 on anything else.

**1. The noise floor is config-dependent; 7e-4 was over-generalised.** Our four depth-14 runs are a clean quadruplicate: sd **2.4e-4**, range 5.2e-4 — 3× tighter than round 0's triplicate. Noise enters via step-count jitter moving where warmdown lands: at 3100 steps a ±50-step wobble is 1.6% of training; at 478 steps it was ±1 step. "Nothing under 1.5e-3" is right for many-step configs, far too lax for few-step ones. At the baseline's ~1100 steps assume sd ≈4e-4.

**2. T [...]
- `#69` **a2**: **Two findings contradict what four of us wrote an hour ago.**

**1. Tokens-per-parameter is NOT the mechanism. Aspect ratio is, and it's bigger.**

a0 (#54), a3 (#50), a4 (#52) and I (#53) all wrote this round up as "tok/param is binding." **I retract that — a0's own run refutes it.**

| shape | aspect | params | tokens | tok/par | val_bpb |
|---|---|---|---|---|---|
| L10 d640 | 64.0 | 85.9M | 293.3M | 3.41 | **0.988076** |
| L10 d768 | 76.8 | 114.8M | 225.2M | 1.96 | 1.018287 |
| L14 d896 | 64.0 | 200.9M | 125.3M | 0.62 | 1.033485 |
| L10 d832 | 83.2 | 130.8M | 186.9M | 1.43 | 1.054503 |

a0's model has **35% fewer params**, saw **49% more tokens**, took 49% more steps — and lost to the depth-14 arm by 0.021 (~90σ). Under a tok/param law it wins easily. The ordering is inverted.

Interpolating the two aspect-64 points in params gives matched-size controls: at 114.8M predicts ~0.999 (a [...]
- `#70` **a3**: **Six runs, baseline moved zero, four were byte-identical.** `md5sum rounds/r01/*/train.py`: a1/a2/a3/a4 all hash `e1257e32b9`. We wrote essays about round 0's duplication, then anti-coordinated into a bigger collision — each of us reasoned "four claimed HEAD_DIM 64, I'll take the unclaimed axis" and picked the same one. **HEAD_DIM 64 was proposed by four agents and run by nobody; it is still not in the baseline.** 12 of 36 runs gone, baseline moved once.

**1. The noise floor is smaller than we've been using.** d14 quadruplicate: 1.033314/1.033354/1.033433/1.033838 — sd **2.4e-4**, vs round 0's 7.1e-4 from n=3. Pooled: **sd 4.8e-4 on 5 dof**. The "nothing under 2e-3" bar (#31, #33) is too strict — 3σ is 1.45e-3 — and would discard real wins of the size we hunt. Spread was 3× tighter at 478 steps than at 3100, so the floor is config-dependent, likely ~2–3e-4 near the baseline's 1100 step [...]
- `#71` **a5**: 
- `#72` **a4**: 

### Experiments run
- `a0/v0` val_bpb **1.054503** (43.6 GB) — ASPECT_RATIO 64->80 and HEAD_DIM 128->64, DEPTH unchanged at 10: n_embd 640->832, 13 heads of 64, 130.8M params. val_bpb 1.054503 vs 0.988076 baseline.
- `a1/v0` val_bpb **1.033354** (33.5 GB) — DEPTH 10->14, HEAD_DIM 128->64, DEVICE_BATCH_SIZE 64->32 (grad_accum 4, same 2^18 total batch); no other changes.
- `a2/v0` val_bpb **1.033433** (33.5 GB) — DEPTH 10->14 and HEAD_DIM 128->64 (n_embd 640->896, 14 heads), DEVICE_BATCH_SIZE 64->32 so grad_accum goes 2->4; TOTAL_BATCH_SIZE and all LRs unchanged.
- `a3/v0` val_bpb **1.033838** (33.5 GB) — DEPTH 10->14 (n_embd 640->896), HEAD_DIM 128->64 (14 heads), DEVICE_BATCH_SIZE 64->32 (grad_accum 2->4, same 2^18 total batch); no other changes.
- `a4/v0` val_bpb **1.033314** (33.5 GB) — DEPTH 10->14 (n_embd 640->896 via ASPECT_RATIO 64), HEAD_DIM 128->64 (14 heads of 64), DEVICE_BATCH_SIZE 64->32 (grad_accum 4) for VRAM headroom; all LRs and schedules untouched.
- `a5/v0` val_bpb **1.018287** (40.4 GB) — ASPECT_RATIO 64 -> 76 and HEAD_DIM 128 -> 64, DEPTH unchanged at 10: 10 layers of dim 768 with 12 heads of 64, vs the baseline's 10 layers of dim 640 with 5 heads of 128.

**Outcome:** nothing beat the baseline; it is unchanged at val_bpb 0.988076.

## Round 2
_Baseline entering the round: val_bpb 0.988076 (commit c4400da)_

### Proposals
- `#75` **a0 — HEAD_DIM 64 at depth 10 — running it this time, unconditionally**
  - why: It is the only confirmed win in the cell (−0.0038, ~8σ, exactly zero cost in params/FLOPs/VRAM), four agents have now proposed it twice and all four peeled off both rounds, so after 12 of 36 runs the baseline still does not contain it.
  - detail: One constant in the hyperparameter block of `train.py`:

- `HEAD_DIM = 128` → `HEAD_DIM = 64`

Nothing else. `DEPTH = 10`, `ASPECT_RATIO = 64`, `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64`, all LRs and schedules untouched.

Mechanism, verified against `build_model_config`: `base_dim = 10*64 = 640`; `model_dim = ceil(640/HEAD_DIM)*HEAD_DIM = 640` either way (640 is divisible by both 64 and 128), so `n_embd` is **unchanged** and `num_heads` goes 5 → 10 (`n_kv_head` follows, still full MHA). Consequences:
- Params: identical. `c_q/c_k/c_v/c_proj` are all `n_embd → n_head*head_dim = n_embd`; `kv_dim = n_kv_head*head_dim = 640` so value embeddings keep their shape. Only `ve_gate` grows `Linear(32,5)`→`Linear(32,10)`, +320 params across 5 layers.
- FLOPs: identical — `estimate_flops` attention term is `12*h*q*window` and `h*q = n_embd` regardless of the split. MFU stays directly comparable to 41.94%.
- VRAM: identical (same tensors, reshaped).
- Muon groups are keyed by `p.shape`, which is unchanged, so optimizer structure is untouched.
- Real changes: the 640-wide residual is partitioned into 10 subspaces instead of 5; rotary is built over `head_dim=64` (32 freq pairs, base 1000 [...]
- `#76` **a1 — HEAD_DIM 128 → 64 at depth 10 (the unrun confirmed win)**
  - why: It is the only measured multi-sigma win in the cell that the baseline still does not contain (a2: −0.0038 at depth 8, ~8σ on the pooled 4.8e-4 floor), it costs exactly zero params/FLOPs/VRAM, and after 12 of 36 runs four agents have proposed it and none has run it.
  - detail: One constant, one line, in the hyperparameter block of `train.py`:

- `HEAD_DIM = 128` → `HEAD_DIM = 64`

Nothing else. `DEPTH = 10`, `ASPECT_RATIO = 64`, `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64` (grad_accum 2), `WINDOW_PATTERN = "SSSL"` and every LR/schedule stay exactly as they are.

Mechanism, checked against `build_model_config`: `base_dim = DEPTH * ASPECT_RATIO = 640`; `model_dim = ceil(640/HEAD_DIM)*HEAD_DIM = 640` for both 64 and 128, so **n_embd is unchanged at 640**; only `num_heads = 640//64 = 10` instead of 5 (and `n_kv_head` follows, so still full MHA). Consequences:

- Params: identical. `c_q/c_k/c_v/c_proj` are all `n_embd → n_head*head_dim = n_embd`; `kv_dim = n_kv_head * head_dim = 640`, so `value_embeds` keep their shape. Only `ve_gate` goes `Linear(32,5) → Linear(32,10)`: +320 params over 5 gated layers, i.e. nothing.
- FLOPs: identical. `estimate_flops`'s attention term is `12*h*q*effective_seq` with `h*q = n_embd` either way, so `num_flops_per_token` and reported MFU stay directly comparable to the 0.988076 baseline.
- VRAM: identical (same activation bytes, different reshape).
- Muon groups are keyed by `p.shape`, which is unchanged, so optimizer gr [...]
- `#74` **a2 — HEAD_DIM 128 → 64 at depth 10 (the unrun free win)**
  - why: It is the only confirmed win in the cell (−0.0038, ~8σ) that the baseline still does not contain, it costs exactly zero params/FLOPs/VRAM, and after 12 runs nobody has actually run it.
  - detail: One constant in the hyperparameter block of `train.py`:

- `HEAD_DIM = 128` → `HEAD_DIM = 64`

Nothing else. `DEPTH = 10`, `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64` (grad_accum 2), all LRs and schedules untouched.

Mechanism, checked against `build_model_config`: `base_dim = DEPTH * ASPECT_RATIO = 640`; `model_dim = ceil(640/HEAD_DIM)*HEAD_DIM = 640` for both 64 and 128, so **n_embd is unchanged at 640** and the model is the same size. Only `num_heads = 640//64 = 10` instead of 5 (`n_kv_head` follows, still full MHA). Consequences:
- Params identical: `c_q/c_k/c_v/c_proj` are all `n_embd → n_head*head_dim = n_embd`; `kv_dim = n_kv_head*head_dim = 640`, so value embeddings keep their shape. Only `ve_gate` goes `Linear(32,5)→Linear(32,10)`, +320 params across 5 gated layers.
- FLOPs identical: `estimate_flops` attention term is `12*h*q*window` with `h*q = n_embd` either way, so reported MFU stays directly comparable to the 0.988076 baseline.
- VRAM identical (same tensors, different reshape).
- Real differences: the 640-wide residual is partitioned into 10 attention subspaces instead of 5; rotary is precomputed over `head_dim=64` (32 freq pairs over base 10000) instead of [...]
- `#78` **a3 — HEAD_DIM 128 → 64 at depth 10 (the unmerged confirmed win)**
  - why: It is the only measured-positive change in the cell (a2: −0.0038, ~8σ on the pooled 4.8e-4 noise floor) that the baseline still does not contain, it is exactly free in params/FLOPs/VRAM, and after 12 of 36 runs it has been proposed six times and run zero times.
  - detail: One-constant change in the hyperparameter block of `train.py`:

- `HEAD_DIM = 128` → `HEAD_DIM = 64` (the line `HEAD_DIM = 128          # target head dimension for attention`).

Nothing else. `DEPTH = 10`, `ASPECT_RATIO = 64`, `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64`, every LR and schedule untouched.

Mechanism, checked against `build_model_config`: `base_dim = DEPTH * ASPECT_RATIO = 640`; `model_dim = ceil(640/HEAD_DIM)*HEAD_DIM = 640` either way (640 is divisible by 64 and by 128), so **n_embd is unchanged at 640**; `num_heads = 640//64 = 10` instead of 5, `n_kv_head` follows (full MHA).

Costs, all zero:
- Params: `c_q/c_k/c_v/c_proj` are all `n_embd → n_head*head_dim = n_embd`; `kv_dim = n_kv_head*head_dim = 640`, so value embeddings keep their shape. Only `ve_gate` goes `Linear(32,5) → Linear(32,10)`: +320 params across 5 gated layers.
- FLOPs: the `estimate_flops` attention term is `12*h*q*window` and `h*q = n_embd` regardless of the split — `num_flops_per_token` and reported MFU stay directly comparable to the 0.988076 baseline.
- VRAM: identical activation shapes, just reshaped.
- Muon groups are keyed by `p.shape`, unchanged, so the optimizer partition is ident [...]
- `#77` **a4 — Merge HEAD_DIM 64 into the depth-10 baseline (10 heads of 64)**
  - why: It is the only confirmed, still-unmerged win in the cell (a2 measured −0.0038, ~8σ on the pooled 4.8e-4 noise floor) and it costs exactly zero params, FLOPs or VRAM — after 12 of 36 runs it has been proposed six times and run zero times at depth 10.
  - detail: One-constant change in the hyperparameter block of `train.py`:

- `HEAD_DIM = 128` → `HEAD_DIM = 64`

Nothing else. `DEPTH = 10`, `ASPECT_RATIO = 64`, `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64` (grad_accum 2) and every LR stay as they are.

Mechanism, checked against `build_model_config`: `base_dim = DEPTH * ASPECT_RATIO = 640`; `model_dim = ceil(640/HEAD_DIM)*HEAD_DIM = 640` either way (640 is divisible by both 64 and 128), so **n_embd is unchanged at 640**; only `num_heads = 640//64 = 10` instead of 5 (`n_kv_head` follows, still full MHA).

Consequences:
- Params: identical. `c_q/c_k/c_v/c_proj` are all `n_embd → n_head*head_dim = n_embd`; `kv_dim = n_kv_head*head_dim = 640`, so value embeddings keep their shape. Only `ve_gate` goes `Linear(32,5)` → `Linear(32,10)`: +160 params × 5 gated layers = +800 total.
- FLOPs: identical. `estimate_flops` uses `attn_flops += 12*h*q*window` and `h*q = n_embd` regardless of the split, so MFU and tok/s stay directly comparable to the 0.988076 baseline.
- VRAM: identical (same tensors, reshaped).
- Real changes: the 640-wide residual is partitioned into 10 subspaces instead of 5; rotary is built at `head_dim=64` (32 frequency pairs ov [...]
- `#79` **a5 — Shrink width: depth 10, dim 512 (+ HEAD_DIM 64)**
  - why: Every size move this cell has tried was growth and every one failed; at 3.4 tokens/param the baseline is wildly over-parameterised for a 5-minute budget, and the one direction the aspect-ratio gradient points to — narrower at fixed depth 10 — has never been sampled.
  - detail: Two constants in the hyperparameter block of `train.py`, nothing else:

- `ASPECT_RATIO = 64` → `ASPECT_RATIO = 51`
- `HEAD_DIM = 128` → `HEAD_DIM = 64`

Keep `DEPTH = 10`, `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64` (grad_accum stays 2), all LRs and schedules untouched.

Arithmetic, checked against `build_model_config`: `base_dim = 10*51 = 510`; `model_dim = ((510+63)//64)*64 = 512` exactly; `num_heads = 512//64 = 8`. So: 10 layers, n_embd 512, 8 heads of 64 — the baseline's depth with the depth-8 arm's width, and the confirmed-free HEAD_DIM 64 split folded in (integers throughout; 51 is chosen so the ceil lands exactly on 512 with no rounding slack).

Expected numbers: ~61M params (wte 4.2M, lm_head 4.2M, value embeds 21M, blocks 31.5M) vs the baseline's 85.9M. FLOPs/token ≈ 6*(61−25.2)M + attn(7*6.29M + 3*12.6M) ≈ 2.97e8 vs the baseline's 4.29e8, i.e. **0.69×** — so ~400M tokens instead of 293M, ~6.6 tokens/param, ~1500 steps. VRAM drops (≈28GB).

Why this and not another growth probe or a fifth HEAD_DIM-64 replicate:
1. The cell has five size points and **all five moves were upward**: d8→d10 won, d10→d12/14 lost, width 640→768→832 lost monotonically. At depth 10 we hav [...]

### Messages
- `#80` **a5**: a5 SLOT CLAIM (#79): five of six proposals this round (#75 a0, #76 a1, #74 a2, #78 a3, #77 a4) are the IDENTICAL one-constant HEAD_DIM 128->64 at depth 10. It will be run five times. Good - it finally gets run, and five replicates hand us a third noise-floor estimate at ~1100 steps, which is genuinely useful. But I am not the sixth copy.

I am running my proposal as written: ASPECT_RATIO 64 -> 51 AND HEAD_DIM 128 -> 64, DEPTH stays 10. base_dim = 510, ceil to a multiple of 64 -> n_embd 512 exactly, 8 heads of 64. ~61M params, ~0.69x FLOPs/token, so ~400M tokens instead of 293M.

Why this is the one slot worth spending differently: every size move the cell has ever made was GROWTH, and every single one failed. d10->d12/14 lost by 0.045, dim 640->768 lost by 0.030, 640->832 lost by 0.066. We have three points on the up-side of the aspect axis at depth 10 (64, 76.8, 83.2) and ZERO on the do [...]
- `#81` **a0**: a0 SLOT CLAIM — LAUNCHING HEAD_DIM 64 AT DEPTH 10 RIGHT NOW. Edit is already in my train.py (line 434, HEAD_DIM = 128 -> 64, nothing else); arena-train goes next.

FIVE of us (#75 me, #76 a1, #74 a2, #78 a3, #77 a4) proposed the identical one-line merge, and only a5 (#79) took a different square. If we all run it, the cell spends 5 of its remaining 24 slots on one number we already estimate at -0.0038. If we all defer to each other, round 1 repeats and it goes unrun for a THIRD round.

So here is a tie-break rule the group can apply mechanically, now and later: **when N agents claim the same square, the LOWEST agent index runs it and everyone else treats it as taken.** No judgement, no negotiation, no reading each other's essays — it resolves without communication, which is the only thing that works when we are writing simultaneously. That is me this round, and I pre-committed to running [...]
- `#82` **a4**: a4 SLOT CLAIM, posted BEFORE launching. Five of six proposals this round are the identical pure HEAD_DIM 64 (#75 a0, #76 a1, #74 a2, #78 a3, #77 me). In #77 I committed in writing to defer if anyone claimed it first; a0 (#75) claimed it 'unconditionally'. I take that at face value and I am NOT running a second copy.

**a1, a2, a3: a0 has pure HD64. Treat it as taken — and DO NOT all flee to the same square again, which is exactly how round 1 burned four slots.** If you defect, say where to, in one line, before you launch.

I am running my stated fallback: **HEAD_DIM 64 + MATRIX_LR 0.04 -> 0.055** (everything else at baseline: DEPTH 10, ASPECT_RATIO 64, 2^18, device batch 64). Two reasons this is the right second slot rather than a replicate:

1. The LR block is the largest untouched region in the cell — 12 of 12 runs went to batch size and model size, zero to optimization. And it is not [...]
- `#83` **a3**: a3 SLOT CLAIM: NOT running pure HEAD_DIM 64. Five of six of us (a0 #75, a1 #76, a2 #74, me #78, a4 #77) proposed the identical one-line change and a0/a1/a2/a4 all said 'unconditionally'. It is covered four times over; a fifth copy buys the cell nothing. Only a5 (#79) is elsewhere, shrinking width to 512.

I am running **HEAD_DIM 64 AND WARMDOWN_RATIO 0.5 -> 0.3**, two constants, nothing else.

Why combine rather than take a pure second axis: the four HD64 arms are a GUARANTEED control this round. (mine - theirs) isolates the warmdown effect exactly, the same structure that produced a2's HD64 win in round 0. So combining costs no interpretability here and, if both parts are positive, it hands the lineage a strictly better baseline in one run instead of two rounds.

Why warmdown specifically. It is the largest untouched knob and it has a real prior, not a hunch. WARMUP_RATIO=0.0 and WARMDO [...]
- `#84` **a1**: a1 SLOT CLAIM (#76 modified): five of six of us proposed the IDENTICAL pure HEAD_DIM 64 (a0 #75, me #76, a2 #74, a3 #78, a4 #77). a0's title says 'running it this time, unconditionally' — I am treating that as the firm claim I asked for in my own notes, so a0 owns the pure arm and it WILL be banked this round.

I am NOT peeling off to an unrelated axis, because that is how we lost round 1. I am running HEAD_DIM 64 **plus** one orthogonal knob, so my arm banks the merge as insurance AND buys a second bit:

  HEAD_DIM = 128 -> 64   (the confirmed free win, a0's arm is my control)
  MATRIX_LR = 0.04 -> 0.06   (Muon LR, +50%)

Everything else untouched: DEPTH 10, ASPECT_RATIO 64, batch 2^18, device batch 64, all Adam LRs, all schedules. (my - a0's) is a clean read on MATRIX_LR, since HEAD_DIM is free in params/FLOPs/VRAM and a0 runs the control.

Why the LR block and why UP, stated before th [...]
- `#85` **a2**: a2 SLOT CLAIM: I am NOT running HEAD_DIM 64. Five of six proposals (#75 a0, #76 a1, #74 me, #78 a3, #77 a4) are that identical one-line change and a0 claimed it unconditionally in #75 — it is covered. I proposed it too (#74) and I am dropping my own proposal.

I am taking a region nobody in this cell has touched in 12 runs: THROUGHPUT AT FIXED MATH. One word:

  model = torch.compile(model, dynamic=False)
  -> model = torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")

Why this is the right bet given what we measured, not just an unexplored square:
1. The batch axis is flat 2^19->2^17 (#15,#17,#32): loss is insensitive to optimizer steps here, so we are NOT optimization-limited.
2. Every size probe failed by starving the model of tokens (#49,#50,#52,#54,#55) while MFU ROSE monotonically with size (37.7 -> 41.9 -> 43.8%).
Both point the same way: in this regime val_bpb [...]
- `#86` **a5** _(re: #80)_: a5 RESULT (#80, width shrink): NEGATIVE. dim 512 at depth 10 (+HD64) -> val_bpb 0.994916 vs 0.988076 baseline, +0.0068, ~14 sigma. 60.8M params, 1439 steps, 377.2M tokens, MFU 37.29%, peak VRAM 28.0GB. I said before the run this was closer to a coin flip than my proposal sounded, and it lost, so: the down-side of the aspect axis is sampled and the SIZE BOX IS CLOSED ON ALL FOUR SIDES. 10 layers x dim 640 beats deeper (d12/14), shallower-and-narrower (d8/512), wider (768, 832) and narrower (512). Nobody should spend another slot on shape.

But the round's real result is an accidental MATCHED PAIR, and it overturns the story the cell has been telling itself for two rounds.

a2 round 0: d8, dim 512, HD64, batch 2^18 -> 0.993500
a5 round 2 (me): d10, dim 512, HD64, batch 2^18 -> 0.994916

Those two configs are identical in every respect except DEPTH. Adding two layers at fixed width is +0.00 [...]
- `#87` **a0** _(re: #81)_: a0 RESULT — **HEAD_DIM 64 IS A REGRESSION, NOT A WIN. val_bpb 0.995347 vs 0.988076 baseline, +0.0073, ~15 sigma.** The one change five of us called 'the confirmed free win' and proposed twice each is measurably bad.

Numbers: 1089 steps, 285.5M tokens, 85.9M params (unchanged, as predicted), MFU 40.81%, peak VRAM 33.9GB, 300.2s. Baseline for comparison: 1119 steps, 293.3M tokens, MFU 41.94%.

**It is not throughput and it is not eval noise.**
- Tokens fell only 2.7% (10 heads of 64 are slightly less efficient through FA3 than 5 of 128 — worth noting, my #75 claimed identical MFU and that was wrong, though only by ~1 point).
- Final smoothed TRAIN loss is 2.788 vs the baseline's 2.752. The model is worse on the data it trained on. A 2.7% token deficit cannot produce that; for calibration, the depth-14 arm lost 57% of its tokens for +0.045.
So ~90%+ of the +0.0073 is representational: spli [...]
- `#88` **a4** _(re: #82)_: a4 RESULT (#82: HEAD_DIM 64 + MATRIX_LR 0.04 -> 0.055): **strong NEGATIVE. val_bpb 1.000971 vs the 0.988076 baseline, +0.0129, ~27 sigma.** I called the direction in advance and I called it wrong.

Numbers: 1085 steps, 284.4M tokens, MFU 40.67%, peak VRAM 33.8GB, final smoothed train loss 2.806 vs the baseline's 2.752, model config n_head 10 / n_embd 640 as intended. No crash, no instability — the loss curve is smooth and monotone, just worse everywhere.

**1. The Muon LR is at or above its optimum, and the effect is large.** My run contains HEAD_DIM 64, so subtract it: whatever the pure-HD64 arm scores (expected ~0.984), a +37% Muon LR alone costs roughly +0.017. That is 3.5x the size of the HEAD_DIM win and 4x any other architectural effect we have measured at fixed size. **The LR axis is live and steep — it is the most sensitive knob found in this cell so far.** The remaining probe is [...]
- `#89` **a3** _(re: #83)_: a3 RESULT (#83, HEAD_DIM 64 + WARMDOWN_RATIO 0.5 -> 0.3): **NEGATIVE, +0.0129.** val_bpb 1.000945 vs the 0.988076 baseline, ~27 sigma on the pooled 4.8e-4 floor.

Numbers: 1091 steps, 286.0M tokens, MFU 40.91%%, peak VRAM 34.7GB, 85.9M params, training_seconds 300.0.

**The negative is the schedule, not the head count.** My run carries HD64, which four of you measured independently this round — subtract your number and the warmdown term is what is left. If HD64 lands near 0.984 as we all predicted, shortening the cooldown cost about **+0.017**. I predicted in #83 that I would call this plainly if it came out worse than the HD64 arms, so: I was wrong about the direction. I argued from the WSD literature that 0.5 sits past the optimum and that 0.3 buys ~10%% more effective LR (average lrm 0.75 -> 0.825) for free. It does exactly that, and it makes the model worse.

**Why this is still the [...]
- `#90` **a1** _(re: #84)_: a1 RESULT (#84: HEAD_DIM 64 + MATRIX_LR 0.04 -> 0.06): NEGATIVE. val_bpb 0.998596 vs the 0.988076 baseline, +0.0105, about 22 sigma on the pooled 4.8e-4 floor. 1091 steps, 286.0M tokens, MFU 40.92%, peak VRAM 34.66GB, 85.9M params, training_seconds 300.0. No crash, clean run.

**1. Muon LR up is wrong, and I was wrong about why it might be right.** In #84 I argued for going UP because my own round-0 run at 2^17 (4x the updates at fixed LR) came out flat rather than unstable, which I read as headroom above 0.04. That inference does not hold. Flat at 4x the updates meant 'already at or past the useful update rate', not 'each update is too small'. The damage here is optimization, not evaluation: final smoothed TRAIN loss 2.817 against the baseline's 2.752, so the model genuinely trained worse, it is not an eval-side artifact.

Since a0 ran pure HEAD_DIM 64 as the control, (mine - a0's) is t [...]
- `#98` **a0** _(re: #85)_: a0 -> a2 (#85): **your round-2 row is not a measurement — max-autotune never ran, and it is still an open, unmeasured axis.**

Evidence, from your own round directory:
- rounds/r02/a2_v0/train.py:508 DOES contain mode="max-autotune-no-cudagraphs", so you made the edit.
- rounds/r02/a2_v0/arena_run.json says {"round": 0, "agent": "a5", ... "workdir": ".../work/r00_a5_v0"} — the artifacts in that directory are the round-0 depth-10 baseline run, carried forward.
- rounds/r02/a2_v0/candidate.json is verbatim the round-0 depth-10 description, and run.log ends at 1119 steps / 41.94% MFU / val_bpb 0.988076, i.e. bit-for-bit the c4400da baseline.
- The budget counter went 24 -> 19 this round: five runs, six agents. Yours is the one that did not spend a slot.

So the log line 'a2/v0 val_bpb 0.988076' should be read as ABANDONED, not as 'max-autotune scored exactly baseline'. If anyone reads it th [...]
- `#99` **a5** _(re: #85)_: a5 -> a2 (#85): your max-autotune run did not happen, and the log's 'a2/v0 val_bpb 0.988076' is not a measurement of your code. Evidence: rounds/r02/a2_v0/arena_run.json reads {"round": 0, "agent": "a5", "workdir": ".../work/r00_a5_v0"} and the candidate.json in that directory is my round-0 baseline text verbatim ('DEPTH 8->10 ... for VRAM headroom'). The 0.988076 is bit-identical to the round-0 baseline, which is impossible for a real replicate given a noise floor of 2-7e-4.

Your rounds/r02/a2_v0/train.py DOES contain the change on line 508: torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs"). So the edit was right; the run is simply missing.

The good news is the budget was not charged for it: 24 -> 19 remaining across six agents is FIVE runs, not six. So max-autotune cost the cell nothing and is still completely untested.

Please re-run it. Your argument in #85 is [...]
- `#100` **a4** _(re: #85)_: a4 -> a2 (#85): **your max-autotune run never happened, and the log is recording a phantom result for it.** Check before you build on it.

Evidence: rounds/r02/a2_v0/arena_run.json reads round 0, agent a5, workdir work/r00_a5_v0 — it is the INHERITED artifact from the baseline commit, not a record of your run. Same for your run.log (1119 steps, 293.3M tokens, val_bpb 0.988076) and your candidate.json, whose title is still 'Depth 10 (dim 640) + batch 2**18' and whose reflection is a5's round-0 text verbatim. Every other agent's rounds/r02/*/arena_run.json says round 2 with their own agent id. The budget confirms it: 24 -> 19 is five runs, and a0/a1/a3/a4/a5 account for all five.

So the round table's line 'a2/v0 val_bpb 0.988076 (33.8 GB)' is the baseline echoed back, not a measurement, and it is NOT evidence that max-autotune-no-cudagraphs ties the baseline. Your train.py does contain th [...]
- `#101` **a1** _(re: #86)_: a1 -> a5 (#86): your width-shrink number is a tie, not a +0.011 loss, and the size box is NOT closed on the narrow side. You subtracted the phantom -0.0038 HD64 'win'. Against the real control a0 ran the same round:

  a0  dim 640 + HD64: val_bpb 0.995347, final train loss 2.7877, 285.5M tokens, 85.9M params, 33.9GB
  a5  dim 512 + HD64: val_bpb 0.994916, final train loss 2.7816, 377.2M tokens, 60.8M params, 28.0GB

You are 4.3e-4 ahead on val_bpb and 0.006 ahead on train loss, with 71% of the parameters, 132% of the tokens and 6GB less memory. Both metrics rank you first, and the val gap is well inside the noise (see my round message - the a0/a1/a4 triple inverts by 2.4e-3 against a monotone train-loss ordering, so sd here is ~1-2e-3, not 4.8e-4).

So the correct statement is: between dim 512 and dim 640 at depth 10, WIDTH IS FLAT. What is not flat is head_dim - a0 measured HD128 -> 64 [...]
- `#102` **a3** _(re: #85)_: a3 -> a2 (#85): your max-autotune run never happened. The slot was spent, but it trained the BASELINE, not your change. Evidence, not inference:

1. md5sum rounds/r02/a2_v0/run.log == rounds/r00/a5_v0/run.log, byte-identical: 47be76b06e5de57bf7a4736836920abc.
2. rounds/r02/a2_v0/arena_run.json literally reads round: 0, agent: a5, workdir .../work/r00_a5_v0. Every other agent's r02 arena_run.json correctly reads round 2 with their own id — I checked all five.
3. Your candidate.json is a verbatim copy of a5's round-0 entry ('DEPTH 8->10 ... TOTAL_BATCH_SIZE 2**19->2**18'), which is why results.tsv row 17 shows 0.988076 with a5's round-0 description.
4. Your train.py DOES have the edit — line 508, mode=max-autotune-no-cudagraphs, and you correctly left HEAD_DIM at 128 — but its mtime is 08:30:14, three minutes AFTER run.log and arena_run.json were stamped at 08:27:06.

So the artifacts were [...]
- `#103` **a3** _(re: #86)_: a3 -> a5 (#86): 'the size box is closed on all four sides' is the one conclusion from this round I think is wrong, and your own run is the evidence against it. You retracted on the obsolete control.

You wrote 'my arm contains HEAD_DIM 64, so the width shrink alone cost roughly +0.011', using a2's -0.0038. a0 ran the real control this round and HD64 is +0.00727, not -0.0038. Subtract the right number:

  a0  HD64, dim 640, depth 10 -> 0.995347
  a5  HD64, dim 512, depth 10 -> 0.994916

Identical in every respect except width. Your narrow model is 0.00043 BETTER. That is a dead tie, not +0.011. Your correction was 0.011 too pessimistic and you talked yourself out of your own result.

And a tie is not nothing, because of what it costs:
  params   60.8M vs 85.9M   (71%)
  tokens   377.2M vs 285.5M (+32%)
  VRAM     28.0GB vs 33.9GB
Equal loss at 71% of the parameters and a third more data. [...]
- `#104` **a2**: **My run did not happen, and three of you subtracted a control that does not exist.**

**1. Ignore the "a2/v0 0.988076" row. It is not a measurement.** My slot never reached the GPU (queued, session ended); the orchestrator harvested the stale baseline `run.log` sitting in my worktree. Tell-tale: it matches the baseline to all six digits *and* to the step count (1119). Budget confirms it — 19 left = 12 + **5**, not 6. So `max-autotune-no-cudagraphs` (#85) is still unmeasured and cost nothing. I am claiming it again next round; nobody else take it.

**2. a4 (#88), a3 (#89): your effect sizes are inflated by ~0.011.** You both subtracted the *predicted* HD64 value (~0.984) instead of a0's *measured* 0.995347. Using the real control:

| arm | vs a0 (HD64 control) |
|---|---|
| MATRIX_LR 0.06 (a1) | **+0.0033** |
| MATRIX_LR 0.055 (a4) | **+0.0056** |
| WARMDOWN 0.3 (a3) | **+0.0056** |

Not [...]
- `#105` **a0**: **My run was the control for four of yours, and it came out +0.0073 instead of −0.0038. Every delta published this round is wrong by that amount.**

Pure HD64 (mine, 42f95bc) = **0.995347**. That is the control, not the imagined ~0.984. Recomputed:

| arm | extra knob | val_bpb | true delta |
|---|---|---|---|
| a0 | — (HD64 only) | 0.995347 | — |
| a1 | MATRIX_LR 0.06 | 0.998596 | **+0.0032** |
| a4 | MATRIX_LR 0.055 | 1.000971 | **+0.0056** |
| a3 | WARMDOWN 0.3 | 1.000945 | **+0.0056** |
| a5 | dim 640→512 | 0.994916 | **−0.0004** |

**1. a3 (#89), a4 (#88): your "+0.017" are ~3× inflated.** Both are +0.0056. a4's "LR is the most sensitive knob in the cell" does not survive: +37% Muon LR costs about what a 0.2 cut in warmdown costs, and both are smaller than HEAD_DIM itself.

**2. The two MATRIX_LR points are non-monotone.** a1 at 0.06 beat a4 at 0.055 by **0.0024** — ~5σ on the 4.8e- [...]
- `#106` **a5**: **I retract the headline of my own result message (#86), and the correction flips its sign.**

In #86 I said "the width shrink alone cost ~+0.011," by subtracting a2's HD64 estimate of −0.0038. a0 (#87) has now measured HD64 against a **real** control: it is **+0.0073**. Subtract the right number:

| arm | config | val_bpb | vs a0's HD64 control |
|---|---|---|---|
| a0 | HD64, dim 640 | 0.995347 | — |
| **a5** | **HD64, dim 512** | **0.994916** | **−0.00043** |
| a1 | HD64, MATRIX_LR 0.06 | 0.998596 | +0.00325 |
| a3 | HD64, WARMDOWN 0.3 | 1.000945 | +0.00560 |
| a4 | HD64, MATRIX_LR 0.055 | 1.000971 | +0.00562 |

a0's arm and mine differ **only** in width. **dim 640 → 512 is free: −0.0004.** Not free-but-pointless: 0.69× FLOPs/token, **+32% tokens (377.2M vs 285.5M)**, −29% params, −6GB. Train loss agrees (2.7816 vs 2.7877).

**So "the size box is closed on all four sides" (my #86) is [...]
- `#107` **a4**: **Three corrections; the first is the one the group needs most: a2 never ran. Its 0.988076 is a phantom row.**

**1. Only five runs happened.** `rounds/r02/a2_v0/arena_run.json` says `round 0, agent a5, workdir work/r00_a5_v0` — inherited artifacts, as are its `run.log` (1119 steps, 293.3M tokens) and `candidate.json`, still titled "Depth 10 + batch 2**18" with a5's round-0 text. Budget agrees: 24→19 = a0/a1/a3/a4/a5. a2's `train.py` *does* have `mode="max-autotune-no-cudagraphs"` — the edit is real, the run is not. **max-autotune is untested, not a tie.** Those stale files sit in every fresh worktree; I nearly misread my own mid-queue.

**2. My #88 arithmetic used the wrong control. Retracting it.** I subtracted an assumed HD64 = −0.0038; the real control is a0's 0.995347. Against it: a4 (LR 0.055) **+0.0056**, a1 (LR 0.06) **+0.0032**, a3 (warmdown 0.3) **+0.0056**. So **+37% Muon LR h [...]
- `#108` **a1**: **1. a2 never trained. `max-autotune` is UNMEASURED — do not read that row as neutral.**

`budget.json` lists five claims this round: a5(13), a0(14), a4(15), a3(16), a1(17). No a2. `rounds/r02/a2_v0/arena_run.json` is literally a5's **round 0** record (`"round": 0, "agent": "a5"`, workdir `r00_a5_v0`), and its `candidate.json` is a5's round-0 candidate. The 0.988076 in a2's row is the inherited baseline, not a measurement. But `a2_v0/train.py:508` *does* carry `mode="max-autotune-no-cudagraphs"` — the edit was made, the run never happened. Throughput-at-fixed-math still has 0 runs out of 17. a2: reclaim it.

**2. The noise floor here is ~1–2e-3, not 4.8e-4. Three runs differing by one character prove it.**

`diff` of a0/a1/a4 `train.py`: identical except `MATRIX_LR` = 0.04 / 0.06 / 0.055.

| MATRIX_LR | train loss | val_bpb |
|---|---|---|
| 0.040 | 2.7877 | 0.995347 |
| 0.055 | 2.8060 | [...]
- `#109` **a3**: **Two things to fix before round 3: a slot that never ran, and every subtraction posted this round (mine included) being wrong by 0.011.**

**1. a2's run never happened — max-autotune is UNMEASURED.** `rounds/r02/a2_v0/run.log` is byte-identical to `rounds/r00/a5_v0/run.log` (md5 `47be76b0`), and its `arena_run.json` reads `round: 0, agent: a5`. All five other r02 files are correct. a2's train.py *does* have the edit, mtime 08:30, three minutes after the artifacts were stamped 08:27. results.tsv row 17 (0.988076) is the r00 baseline, not a measurement. **The round had 5 arms, not 6. Drop row 17 from any noise-floor arithmetic, and throughput-at-fixed-math is still untested** (#102).

**2. HD64 is +0.00727, so recompute.** Against a0's real control:

| arm | Δ baseline | Δ HD64 ctrl |
|---|---|---|
| a0 HD64 | +0.00727 | — |
| a3 WARMDOWN 0.3 | +0.01287 | **+0.00560** |
| a4 MATRIX_LR .05 [...]

### Experiments run
- `a0/v0` val_bpb **0.995347** (33.8 GB) — HEAD_DIM 128 -> 64 at DEPTH 10: 10 heads of 64 instead of 5 of 128, n_embd unchanged at 640, nothing else touched.
- `a1/v0` val_bpb **0.998596** (33.8 GB) — HEAD_DIM 128->64 (10 heads of 64 at unchanged n_embd 640) plus MATRIX_LR 0.04->0.06; everything else untouched.
- `a2/v0` val_bpb **0.988076** (33.8 GB) — DEPTH 8->10 (n_embd 512->640, 5 heads), TOTAL_BATCH_SIZE 2**19->2**18, DEVICE_BATCH_SIZE 128->64 (grad_accum 2) for VRAM headroom.
- `a3/v0` val_bpb **1.000945** (33.8 GB) — Two constants at depth 10: HEAD_DIM 128->64 (10 heads of 64, n_embd unchanged at 640) and WARMDOWN_RATIO 0.5->0.3; nothing else.
- `a4/v0` val_bpb **1.000971** (33.8 GB) — HEAD_DIM 128->64 (10 heads of 64 at unchanged n_embd 640) plus MATRIX_LR 0.04->0.055; DEPTH 10, ASPECT_RATIO 64, TOTAL_BATCH_SIZE 2**18, DEVICE_BATCH_SIZE 64 and all other LRs unchanged.
- `a5/v0` val_bpb **0.994916** (27.4 GB) — ASPECT_RATIO 64->51 (n_embd 640->512, 8 heads) and HEAD_DIM 128->64 at DEPTH 10; 60.8M params, 0.69x FLOPs/token, 377M tokens in 1439 steps.

**Outcome:** nothing beat the baseline; it is unchanged at val_bpb 0.988076.

## Round 3
_Baseline entering the round: val_bpb 0.988076 (commit c4400da)_

### Proposals
- `#114` **a0 — WARMDOWN_RATIO 0.5 → 0.7 (longer cooldown) on the true HD128 baseline**
  - why: Round 2's only two coherent signals — shortening the cooldown (+0.0056) and raising Muon LR (+0.0032/+0.0056) — both say the effective LR is too high late in training, and the mirror probe of a3's measured warmdown slope is the cheapest way to act on that: one constant, zero FLOPs/params/VRAM, same [...]
  - detail: One constant in the hyperparameter block of `train.py`:

- `WARMDOWN_RATIO = 0.5` → `WARMDOWN_RATIO = 0.7`

Nothing else. Explicitly keep `HEAD_DIM = 128` (a0's round-2 run measured HD64 at **+0.00727**, ~15σ — it is a regression, not the "free win" five of us proposed twice; nobody should carry it this round), `DEPTH = 10`, `ASPECT_RATIO = 64`, `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64`, `MATRIX_LR = 0.04`, all other LRs, `WARMUP_RATIO = 0.0`, `FINAL_LR_FRAC = 0.0`.

Mechanism. `get_lr_multiplier(progress)` with `WARMUP_RATIO = 0` is: `lrm = 1.0` while `progress < 1 - WARMDOWN_RATIO`, then linear decay to `FINAL_LR_FRAC = 0`. So the constant-LR phase shrinks from the first 50% of the time budget to the first 30%, and the ramp to zero stretches over the last 70%. Time-average `lrm` goes 0.75 → 0.65. `progress` keys off `total_training_time / TIME_BUDGET`, so this is exact in wall-clock and needs no step-count assumption. It applies uniformly to every param group (all AdamW and all Muon groups get `initial_lr * lrm`), so it lowers the *whole* optimizer's late step size in a shape-preserving way rather than re-weighting Muon against Adam — which is what a MATRIX_LR cut woul [...]
- `#113` **a1 — Lengthen the LR warmdown: WARMDOWN_RATIO 0.5 → 0.7**
  - why: Every LR-magnitude probe in this cell points the same way — more effective LR is worse (MATRIX_LR 0.055/0.06 lost 0.006/0.003; warmdown 0.3, which raises average lrm, lost 0.0056) — so a longer decay is the one untested direction on the only axis with a consistent measured gradient, and it is free i [...]
  - detail: One constant in the hyperparameter block of `train.py`:

- `WARMDOWN_RATIO = 0.5` → `WARMDOWN_RATIO = 0.7`

Nothing else. `DEPTH = 10`, `ASPECT_RATIO = 64`, `HEAD_DIM = 128` (5 heads of 128 — the baseline's, NOT 64, which a0 measured at +0.0073), `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64` (grad_accum 2), `WARMUP_RATIO = 0.0`, `FINAL_LR_FRAC = 0.0`, `MATRIX_LR = 0.04` and all Adam LRs untouched.

Mechanism. `get_lr_multiplier(progress)` with `WARMUP_RATIO = 0` holds `lrm = 1.0` until `progress = 1 - WARMDOWN_RATIO`, then decays linearly to `FINAL_LR_FRAC = 0`. At 0.5 the decay starts halfway and the time-average multiplier is 1 - 0.5/2 = **0.75**; at 0.7 it starts at 30% and the average is **0.65** — a 13% cut in average LR *plus* a gentler final approach to zero. Both components run in the direction the cell's data supports:

- a3 (#89) went the other way (0.3, average lrm 0.825) and lost +0.0056 against a0's matched HD64 control.
- a1 (#90) +50% MATRIX_LR: +0.0032. a4 (#88) +37%: +0.0056. Both raised the effective step size, both lost, and both had *worse final train loss* (2.817 / 2.806 vs the control's 2.788) — the damage is optimization, not evaluation. The model is o [...]
- `#115` **a2 — Longer warmdown: WARMDOWN_RATIO 0.5 → 0.7**
  - why: Two independent round-2 measurements (shorter warmdown +0.0056, higher Muon LR +0.003–0.006) both say the effective LR is too high for the halved 2^18 batch, and a longer cooldown is the cheapest untested way to lower it while improving the final anneal.
  - detail: One constant in the hyperparameter block of `train.py`:

- `WARMDOWN_RATIO = 0.5` → `WARMDOWN_RATIO = 0.7`

Nothing else. `DEPTH = 10`, `ASPECT_RATIO = 64`, **`HEAD_DIM` stays 128** (a0 measured HD64 at +0.0073 — it is a regression, not the "free win" five of us believed), `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64`, all LRs, `WEIGHT_DECAY`, betas and `FINAL_LR_FRAC` untouched. Clean single-variable delta against the 0.988076 baseline; no control run needed.

Mechanism, from `get_lr_multiplier(progress)` with `WARMUP_RATIO = 0.0`, `FINAL_LR_FRAC = 0.0`:
- now: lrm = 1.0 until progress 0.5, then linear 1→0 over the last half. Mean lrm 0.75.
- after: lrm = 1.0 until progress 0.3, then linear 1→0 over the last 70%. Mean lrm 0.65. Continuous at the junction (0.7/0.7 = 1.0), still exactly 0 at progress 1.
Zero cost: no params, no FLOPs, no VRAM, same steps, same tokens. `progress` is wall-clock based so the schedule lands correctly regardless of throughput jitter.

Why this knob and this direction:
1. a3 (#89) ran the opposite move (0.5 → 0.3) and it cost **+0.0056** against a0's real HD64 control — the largest one-knob derivative measured at fixed model size in this cell, and [...]
- `#112` **a3 — Lengthen the LR warmdown (0.5 → 0.7)**
  - why: Round 2's two independent negatives — Muon LR up (+0.003/+0.006) and warmdown cut to 0.3, i.e. a higher average LR (+0.0056) — both say the effective LR is at or above its optimum, and lengthening warmdown walks down that measured gradient for every parameter group at once, for one constant and zero [...]
  - detail: One constant in the hyperparameter block of `train.py`:

- `WARMDOWN_RATIO = 0.5` → `WARMDOWN_RATIO = 0.7`

Nothing else. `DEPTH = 10`, `ASPECT_RATIO = 64`, **`HEAD_DIM` stays 128** (a0 measured HD64 at +0.0073 — it is a regression, do not fold it in), `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64`, all LRs, `WARMUP_RATIO = 0.0` and `FINAL_LR_FRAC = 0.0` untouched.

Mechanism, from `get_lr_multiplier(progress)`: the plateau lasts `1 - WARMDOWN_RATIO` of the time budget, then `lrm` decays linearly to `FINAL_LR_FRAC = 0`. The trajectory goes from {half the budget at lrm=1, then linear decay; mean lrm 0.75} to {30% at lrm=1, then a longer decay; mean lrm 0.65}. That is a ~13% cut in time-averaged LR applied uniformly to all six groups, since the loop sets `group["lr"] = group["initial_lr"] * lrm` for every one of them — Muon matrices, wte, value embeds, lm_head and both scalar groups. `get_weight_decay` is unaffected (keys off `progress`), as is `get_muon_momentum` (keys off `step`).

Zero cost: no parameter, FLOP, VRAM or throughput change, so `num_steps`, tokens and MFU should land on the baseline's 1119 / 293.3M / 41.9% and any val_bpb delta is purely the schedule.

Why this [...]
- `#116` **a4 — Longer LR warmdown: WARMDOWN_RATIO 0.5 → 0.7**
  - why: Two measurements this cell (shorter warmdown +0.0056; Muon LR up +0.003–0.006) both say the run is over-stepped, and lengthening the decay is the untested good direction on the same knob — free in FLOPs, tokens and VRAM.
  - detail: One constant in the hyperparameter block of `train.py`:

- `WARMDOWN_RATIO = 0.5` → `WARMDOWN_RATIO = 0.7`

Nothing else. `DEPTH = 10`, `ASPECT_RATIO = 64`, `HEAD_DIM = 128` (keep it — a0 measured 64 at +0.0073), `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64`, `MATRIX_LR = 0.04`, all Adam LRs, `WEIGHT_DECAY = 0.2`, `WARMUP_RATIO = 0.0` unchanged.

Mechanism. `get_lr_multiplier(progress)` holds `lrm = 1.0` until `progress = 1 - WARMDOWN_RATIO`, then decays linearly to `FINAL_LR_FRAC = 0.0`. At 0.7 the plateau ends at `progress = 0.3` and the decay covers the last 70% of the budget. Two consequences, both in the direction the data points:
1. Average LR multiplier drops 0.75 → 0.65 (−13% effective LR) without touching any LR constant, so it does not risk early-training instability the way a raw LR cut would.
2. Far more of the budget is spent annealing, which is what converts the last steps into val loss at a short horizon.

Why I believe the direction. `WARMDOWN_RATIO` is one of only two knobs with a measured slope: a3 (#89) ran 0.5 → 0.3 and lost **+0.0056** against a0's matched HD64 control. That gradient points at *more* decay and nobody sampled that side. Independently, a1 [...]
- `#111` **a5 — WARMDOWN_RATIO 0.5 → 0.7 (longer LR decay)**
  - why: Every LR-magnitude probe this cell has run says the schedule is too hot (Muon LR +37% cost +0.0056, and shortening warmdown to 0.3 — which *raises* average LR — cost +0.0056), so the one untested direction on that gradient is a longer decay, which lowers average LR across all param groups in one con [...]
  - detail: One constant in the hyperparameter block of `train.py`:

- `WARMDOWN_RATIO = 0.5` → `WARMDOWN_RATIO = 0.7`

Nothing else. `DEPTH = 10`, `ASPECT_RATIO = 64`, `HEAD_DIM = 128` (5 heads of 128 — a0 measured HD64 as +0.0073, so it stays at 128), `TOTAL_BATCH_SIZE = 2**18`, `DEVICE_BATCH_SIZE = 64`, `MATRIX_LR = 0.04`, all Adam LRs, `WEIGHT_DECAY = 0.2`, `FINAL_LR_FRAC = 0.0` untouched.

Mechanism. `get_lr_multiplier(progress)` holds `lrm = 1.0` until `progress = 1 - WARMDOWN_RATIO`, then decays linearly to `FINAL_LR_FRAC = 0`. At 0.5 the constant phase is the first half and the time-average multiplier is 0.75; at 0.7 the constant phase ends at 30% and the average is 0.65 — a ~13% cut in the LR integral, applied uniformly to every group (Muon matrices, wte, value embeds, lm_head, scalars) rather than to one of them. Zero cost: no params, no FLOPs, no VRAM, no throughput change, so token count and MFU stay directly comparable to the baseline's 1119 steps / 293.3M tokens / 41.94% MFU. Note `get_weight_decay(progress) = WEIGHT_DECAY*(1-progress)` is independent of this and is unchanged.

Why this and not another LR knob. Two arms last round moved effective LR up and both lost against a0's [...]

### Messages
- `#117` **a1**: a1 SLOT CLAIM, posted BEFORE launching: **all SIX of us proposed the identical WARMDOWN_RATIO 0.5 -> 0.7** (#114 a0, #113 me, #115 a2, #112 a3, #116 a4, #111 a5). That is round 0 and round 1 happening a third time, and we have 19 runs left. I am NOT running 0.7.

I pre-committed in my proposal notes (#113) to exactly this contingency: **if two or more agents claim 0.7, I take WARMDOWN_RATIO = 1.0 instead.** So that is what I am running, and I am saying it before the GPU, not after.

Config: WARMDOWN_RATIO = 0.5 -> **1.0**, one constant, nothing else. HEAD_DIM stays 128, DEPTH 10, ASPECT_RATIO 64, TOTAL_BATCH_SIZE 2^18, DEVICE_BATCH_SIZE 64, MATRIX_LR 0.04, WARMUP_RATIO 0.0, FINAL_LR_FRAC 0.0. With WARMDOWN_RATIO=1.0 the plateau disappears entirely: get_lr_multiplier returns cooldown = (1-progress)/1.0, i.e. **lrm = 1 - progress**, pure linear decay to zero from the first step. Time-avera [...]
- `#118` **a0**: a0 SLOT CLAIM — LAUNCHING WARMDOWN_RATIO 0.5 -> 0.7 NOW. Edit is in (train.py:446), arena-train goes next.

ALL SIX of us proposed the identical one-constant change (#114 me, #113 a1, #115 a2, #112 a3, #116 a4, #111 a5). Six replicates would spend 6 of 19 remaining slots on one number.

Apply the tie-break rule from #81: **when N agents claim the same square, the LOWEST agent index runs it and everyone else treats it as taken.** That is me. a1/a2/a3/a4/a5: warmdown 0.7 is COVERED — do not run it.

Note I am overriding my own #114 caveat ('I will drop this if two or more claim warmdown-up'). That caveat was written for a crowded-but-not-unanimous round; applied here it produces exactly the round-1 catastrophe, where four of us each deferred to the others and the square went unrun while we all collided somewhere else. A mechanical rule beats a polite one. The unanimity also means there is [...]
- `#119` **a5**: a5 SLOT CLAIM — posted BEFORE launching. All SIX of us proposed the identical WARMDOWN_RATIO 0.5 -> 0.7 (#114 a0, #113 a1, #115 a2, #112 a3, #116 a4, #111 me). That is a 6x collision, worse than round 0. Under the tie-break rule a0 proposed in #81 and everyone praised — lowest index runs it, everyone else treats it as taken — a0 owns warmdown 0.7. I am NOT running it.

I am running the orthogonal half of the same hypothesis: GLOBAL LR SCALE x0.75, schedule untouched.

  EMBEDDING_LR   0.6   -> 0.45
  UNEMBEDDING_LR 0.004 -> 0.003
  MATRIX_LR      0.04  -> 0.03
  SCALAR_LR      0.5   -> 0.375

WARMDOWN_RATIO stays 0.5, HEAD_DIM stays 128, DEPTH 10, ASPECT_RATIO 64, batch 2^18, device batch 64, WEIGHT_DECAY 0.2, betas unchanged. One multiplicative factor on four constants; zero cost in params/FLOPs/VRAM/tokens.

Why this and not a seventh warmdown copy:

1. Warmdown 0.7 and a global LR cut [...]
- `#120` **a2**: a2 SLOT CLAIM, posted BEFORE launching. ALL SIX of us proposed the identical WARMDOWN_RATIO 0.5 -> 0.7 (#114 a0, #113 a1, #115 me, #112 a3, #116 a4, #111 a5). Six copies of one constant out of 19 remaining runs would be the worst allocation this cell has made. a0 owns it under the lowest-index rule from #81. I am NOT running a second copy of my own proposal.

I am running MATRIX_LR 0.04 -> 0.028 at the true baseline (HEAD_DIM stays 128, WARMDOWN stays 0.5, DEPTH 10, 2^18, device batch 64). One constant, nothing else.

Why this specific defection, and why NOT the combination:
1. Same hypothesis as warmdown-0.7 ('the run is over-stepped'), different knob, measured INDEPENDENTLY against the 0.988076 baseline that everyone already has as a control. If the hypothesis is right, the round returns two separate effect sizes instead of one number six times.
2. I deliberately did NOT combine it wit [...]
- `#121` **a3**: a3 SLOT CLAIM, posted BEFORE launching. **All SIX of us proposed the identical WARMDOWN_RATIO 0.5 -> 0.7** (#114 a0, #113 a1, #115 a2, #112 me, #116 a4, #111 a5). That is the round-0/round-2 failure mode at full strength. By a0's own tie-break rule (#81, lowest index runs the contested square) **a0 owns pure 0.7**; a2 pre-claimed max-autotune in #104. So the two claimed squares are taken and the rest of us must land on FOUR DIFFERENT ones. a1/a4/a5: the obvious defections are WARMDOWN 0.85 and MATRIX_LR 0.03 — please say which in one line before you launch, and do not all take the same one, which is exactly how round 1 burned four slots on depth 14.

**I am running: WARMDOWN_RATIO 0.5 -> 0.7 AND WINDOW_PATTERN "SSSL" -> "L" (full 2048 context on all 10 layers).** Two constants, nothing else. HEAD_DIM stays 128.

Why combine rather than take a pure orthogonal axis: a0's 0.7 arm is a GUARA [...]
- `#122` **a4**: a4 SLOT CLAIM, posted BEFORE launching. ALL SIX of us proposed the identical WARMDOWN_RATIO 0.5 -> 0.7 (#111 a5, #112 a3, #113 a1, #114 a0, #115 a2, #116 me). Six replicates would be the worst round yet. By a0's tie-break rule (#81, lowest index runs it) a0 owns warmdown 0.7 - I am NOT running it, and a2 pre-claimed max-autotune in #104, so that is a2's.

I am ALSO dropping my own announced fallback (MATRIX_LR 0.04 -> 0.028), and I want the reason on the record because it applies to everyone defecting right now: 'LR up hurt, so LR down helps' does not follow. a4 at 0.055 (+0.0056) and a1 at 0.06 (+0.0032) only establish that the gradient at 0.04 points uphill going up. A symmetric optimum sitting exactly at 0.04 predicts the same data, and predicts that -30% costs about as much as +37% did. Same logic applies to warmdown: 0.3 being worse is equally consistent with 0.5 being optimal. a0's [...]
- `#123` **a1** _(re: #117)_: a1 RESULT (#117, WARMDOWN_RATIO 0.5 -> 1.0, pure linear decay): **val_bpb 0.989041 vs the 0.988076 baseline. +0.00097 — a tie.**

Numbers, and they make this an unusually clean comparison: **1120 steps** (baseline 1119), **293.6M tokens** (293.3M), MFU 42.00% (41.94%), peak VRAM 34.6GB, 300.1s. Throughput is matched to within 0.1%, so nothing here is a tokens/sec artifact. Final smoothed train loss 2.759 vs the baseline's 2.752 — val and train agree.

**The headline for the five of you who ran or are running 0.7: the warmdown knob is flat on the long side, and my arm is the STRONGER version of your change.**

I cut the LR integral by 33% (mean lrm 0.75 -> 0.50). 0.7 cuts it by 13% (0.75 -> 0.65). The larger move bought nothing. So the shape of the knob is asymmetric:

| WARMDOWN | mean lrm | result |
|---|---|---|
| 0.3 | 0.825 | **+0.0056** vs matched HD64 control (a3, #89) |
| 0.5 | 0. [...]
- `#124` **a0** _(re: #118)_: a0 RESULT (#118, WARMDOWN_RATIO 0.5 -> 0.7): **val_bpb 0.987227 vs 0.988076. That is -0.00085 — a marginal win at best, and I am not calling it confirmed.**

It is ~1.8 sigma on the pooled 4.8e-4 floor and well inside the ~1-2e-3 floor a1 argues for in #108. One run cannot separate it from noise. If five of you also ran this square, pool the numbers and report ONE value with a spread — that is worth far more than five agents each claiming their own draw.

Numbers, and the throughput is the reason this is a clean read: 1121 steps (baseline 1119), 293.9M tokens (293.3M), MFU 42.03% (41.94%), 34.6GB, 85.9M params, 300.1s. Identical work done; the only difference is where the LR sat.

**The one genuinely informative detail: train and val moved in OPPOSITE directions.** Final smoothed train loss 2.765 vs the baseline's 2.752 — the model fit its training data slightly WORSE and generalised sli [...]
- `#125` **a5**: a5 RESULT (#119, global LR scale x0.75): **NULL. val_bpb 0.988598 vs the 0.988076 baseline, +0.00052 — inside the noise floor.** And it is the cleanest null the cell has produced, because every other number matched the baseline almost exactly:

| | a5 (LR x0.75) | baseline |
|---|---|---|
| val_bpb | 0.988598 | 0.988076 |
| steps | 1120 | 1119 |
| tokens | 293.6M | 293.3M |
| MFU | 41.99% | 41.94% |
| final train loss | 2.7566 | 2.752 |
| peak VRAM | 34.6GB | 34.6GB |

Same steps, same tokens, same throughput, same train loss. The learning rates were the only thing that differed, and cutting ALL FOUR of them by 25% moved nothing.

**1. The LR basin is asymmetric, and the group's shared inference from round 2 was wrong — mine included.**

We had three up-moves, all losses: Muon +37% -> +0.0056, Muon +50% -> +0.0032, warmdown 0.3 (+10% average lrm) -> +0.0056. Six of us read that as 'the r [...]
- `#126` **a2** _(re: #120)_: a2 RESULT (#120, MATRIX_LR 0.04 -> 0.028): DEAD TIE. val_bpb 0.988612 vs the 0.988076 baseline, +0.00054. Run verified real this time (arena_run.json: round 3, agent a2) - unlike my phantom row last round.

Numbers: 1119 steps, 293.3M tokens, MFU 41.98%, peak VRAM 34.6GB, 85.9M params, final smoothed train loss 2.7538 vs the baseline's 2.752. Step count, tokens, MFU and VRAM are IDENTICAL to the baseline to three digits, so this is the cleanest matched pair in the cell - no throughput confound, no shape confound, one constant.

**The Muon LR axis is closed, and the shape of it matters more than the tie.** Three points around 0.04 now:

  0.028 (-30%) -> +0.0005   (me)
  0.040        -> 0         (baseline)
  0.055 (+37%) -> +0.0056   (a4, HD64-corrected)
  0.060 (+50%) -> +0.0032   (a1, HD64-corrected)

Sharp on the high side, flat on the low side. That is the textbook asymmetric LR curv [...]
- `#127` **a3** _(re: #121)_: a3 RESULT (#121: WARMDOWN_RATIO 0.7 + WINDOW_PATTERN "L"): **val_bpb 0.989734 vs the 0.988076 baseline, +0.0017.** By my own pre-registered bar (under 0.002 from n=1 is a tie) that is a TIE on the wrong side, not a win and not a real regression. I am not going to dress it up either way.

Numbers: 1096 steps, 287.3M tokens, **MFU 46.38%**, peak VRAM 33.8GB, 85.9M params, FLOPs/token 4.84e8, final smoothed train loss 2.774 (baseline 2.752), 300.0s.

**The throughput finding is the real result of this run, and it is big enough to change how the group reasons about FLOPs.**

I predicted all-L would cost ~11% of the tokens, because it raises attention FLOPs 12.8% (12*640*13312 -> 12*640*20480, total 4.29e8 -> 4.84e8). **It cost 2.0%** — 287.3M tokens vs 293.3M, 1096 steps vs 1119. MFU went 41.94% -> **46.38%**, the highest number anyone has posted in this cell, beating a5's width-768 43.84% a [...]
- `#128` **a4** _(re: #122)_: a4 RESULT (#122, attention logit scale 0.0884 -> 0.12): NEUTRAL, and I am calling it flat rather than dressing it up. val_bpb 0.988490 vs the 0.988076 baseline = +0.00041, i.e. well inside the noise floor by anyone's estimate of it (2.4e-4 quadruplicate, 7e-4 triplicate, a1's ~1e-3).

Numbers: 1111 steps, 291.2M tokens, MFU 41.66%, peak VRAM 34.6GB, 300.1s, 85.9M params. Zero throughput cost exactly as predicted, so the comparison is clean.

The detail that stops me calling it a tie-that-might-be-a-win: final smoothed TRAIN loss 2.7617 against the baseline's 2.7519. Both metrics agree on the sign, and train loss is the lower-variance of the two, so the honest summary is 'flat, if anything a hair worse'. There is no masked win here.

Why I think it is flat, since the prior was decent (modded-nanogpt sets attn_scale 0.12 at exactly head_dim 128 with exactly this QK-RMSNorm setup): this mod [...]
- `#136` **a0** _(re: #127)_: a0 -> a3 (#127): your run gives the cleanest number in the round and I do not think you extracted it. My arm is your exact control — both carry WARMDOWN 0.7, HEAD_DIM 128, everything else baseline — so the window term is (yours - mine) = 0.989734 - 0.987227 = **+0.00251** for 'SSSL' -> 'L'. Round-3 arm-to-arm sd is ~8e-4, so that is ~3 sigma: the only effect anyone measured this round that is plausibly outside noise, and it is the one you called a tie because you compared against the baseline instead of me.

Sign it properly: MORE attention context made this model WORSE, by more than any LR knob we touched. Combined with your own throughput finding (all-L cost 2.0% of tokens, not the 11% the FLOP count predicted, MFU 41.94 -> 46.38), the useful conclusion is not 'FLOPs are cheap'. It is that the gradient on the window axis points DOWN and nobody has sampled that side. The cheap version o [...]
- `#137` **a2** _(re: #127)_: a2 -> a3 (#127): you graded your own run against the wrong control and talked yourself out of the round's most interesting non-LR number.

I diffed your train.py against a0's: they differ in EXACTLY ONE character range, WINDOW_PATTERN 'SSSL' vs 'L'. Same warmdown 0.7, same HEAD_DIM 128, same everything else. So a0 is your matched control, not the 0.988076 baseline, and the window term is:

  a0  SSSL + wd0.7 -> 0.987227
  a3  L    + wd0.7 -> 0.989734
  window term = +0.002507

You reported +0.0017 against the baseline and called it 'a tie on the wrong side' by your pre-registered 0.002 bar. Against the correct control it is +0.0025, which CLEARS your own bar - and it is the largest effect anyone measured this round outside the LR block, roughly 3.5 sigma on the 7e-4 floor. Train loss agrees and is the lower-variance metric: 2.7743 vs a0's 2.7654, the worst of the six arms.

So your run i [...]
- `#138` **a3** _(re: #124)_: a3 -> a0 (#124): your 'train and val moved in OPPOSITE directions' is an artifact of the metric, and fixing it makes your win look BETTER, not worse.

The final smoothed train loss is a single EMA sample with a ~10-step window. I measured its wobble over the last 40 steps of four different runs, where lrm is ~0 and the model is barely moving: sd 0.0061-0.0071, range 0.021-0.025 (r03/a0, r03/a1, r03/a5, r00/a5). The entire train-loss spread of round 3 (2.7538 to 2.7743) fits inside the wobble of any ONE arm. So 2.765 vs 2.752 is not a measurement of anything.

Average the last 60 logged values instead and the ranking inverts:

  arm        last-value   mean-last-60
  a0 (0.7)   2.7654 (6th)  **2.7625 (1st)**
  a4         2.7617        2.7655
  a1 (1.0)   2.7593        2.7655
  a5         2.7566        2.7661
  baseline   2.7519 (1st)  2.7664 (5th)
  a2         2.7538        2.7680
  a3 (m [...]
- `#139` **a2**: **The round measured one thing well: the optimization block is FLAT. The winner is a noise draw, and building on it will burn the last 13 runs.**

**1. Four LR cuts spanning 0.67×–1.0× of the LR integral all landed in the same ±0.001 band — ordering inverted.**

| arm | mean lrm | Muon LR | val_bpb | Δ | train |
|---|---|---|---|---|---|
| baseline | 0.75 | 0.040 | 0.988076 | — | 2.7519 |
| **a0** wd 0.7 | 0.65 | 0.040 | **0.987227** | **−0.00085** | 2.7654 |
| a5 all-LR ×0.75 | 0.75 | 0.030 | 0.988598 | +0.00052 | 2.7566 |
| a2 MATRIX_LR .028 | 0.75 | 0.028 | 0.988612 | +0.00054 | 2.7538 |
| a1 wd 1.0 | 0.50 | 0.040 | 0.989041 | +0.00097 | 2.7593 |

sd of those deltas = **7.9e-4**. Round 0's sd from running *byte-identical code* was 7.1e-4. **Changing the LR integral by a third scatters exactly as much as changing nothing.** a0's win is the *smallest* cut of the four; a1's −33% should h [...]
- `#140` **a5**: 
- `#141` **a0**: **I won this round and the win is almost certainly noise.**

**1. Best-of-six on a flat surface gives exactly my number.** Six *different* interventions landed in a 0.0025 band: 0.987227 / 0.988490 / 0.988598 / 0.988612 / 0.989041 / 0.989734. Pooled with baseline: mean 0.988540, **sd 7.8e-4**. Expected min of 6 draws under pure noise = mean − 1.267σ = **0.987555**. I got 0.987227. WARMDOWN 0.7 is not a −0.00085 effect; I said so in #124 before I knew I'd won.

**2. The lineage advanced on an order statistic — plan for it.** The new baseline is ~0.0013 optimistic, so round 4's arms will *systematically* read ~0.001 worse than they are and the round will look like "six more regressions" when it may not be. **Compare to the round-3 pool (0.9885 ± 0.0008), not to 0.987227**, or we ratchet on noise and start rejecting real 0.002 wins.

**3. Our shared hypothesis is refuted.** Five arms, four [...]
- `#142` **a1**: **Six distinct configs, one distinct hypothesis — and the spread is indistinguishable from noise.**

We fixed duplication: first round with six different `train.py`. It bought almost nothing, because we deconflicted on the *knob*, not the *hypothesis*. Five of six arms (warmdown 0.7/1.0, MATRIX_LR −30%, LRs ×0.75, warmdown+window) tested one claim — "the run is over-stepped" — and all five came back null.

**1. The region is flat.** a0 0.987227 / a4 0.988490 / a5 0.988598 / a2 0.988612 / a1 0.989041 / a3 0.989734. Mean 0.988617, **sd 8.2e-4**, against replicate noise sd 4.8e-4 pooled (7.1e-4 in r00). Six configs spanning LR ±30%, the LR integral 0.75→0.50, attention logits ×1.36 and full context on every layer gave a between-config spread barely above the within-config one. **No arm this round has a detectable effect, either sign.**

**2. a0's win is what noise predicts.** E[min of 6 dra [...]
- `#143` **a4**: **Six different changes, six flat results. Round 3 measured the noise floor, not the knobs.**

| arm | change | train loss | val_bpb |
|---|---|---|---|
| baseline | — | 2.7519 | 0.988076 |
| a2 | MATRIX_LR −30% | 2.7538 | 0.988612 |
| a5 | all LRs ×0.75 | 2.7566 | 0.988598 |
| a1 | warmdown 1.0 | 2.7593 | 0.989041 |
| a4 | attn scale 0.12 | 2.7617 | 0.988490 |
| **a0** | **warmdown 0.7** | **2.7654** | **0.987227** |
| a3 | warmdown 0.7 + all-L | 2.7743 | 0.989734 |

Mean 0.988617, **sd 8.2e-4** across six *distinct* changes whose train losses span 0.022 nats — an upper bound on our noise.

**1. a0: your −0.00085 is ~1σ of that, and your arm has the second *worst* train loss of the seven with the best val.** Spearman between train and val rank across all seven is **0.21** — no signal. You said as much in #124; here is the number. Keeping 0.7 costs nothing, but **nobody probe 0.85**: 0.3 [...]
- `#144` **a3**: 

### Experiments run
- `a0/v0` val_bpb **0.987227** (33.8 GB) — One constant: WARMDOWN_RATIO 0.5 -> 0.7 at the unmodified depth-10 / HEAD_DIM 128 baseline; LR plateau ends at 30% of the budget instead of 50%, time-average lrm 0.75 -> 0.65. Nothing else changed.
- `a1/v0` val_bpb **0.989041** (33.8 GB) — One constant: WARMDOWN_RATIO 0.5 -> 1.0, so the LR plateau disappears and lrm = 1 - progress decays linearly to zero from step 0; time-average lrm 0.75 -> 0.50. Nothing else changed (HEAD_DIM 128, DEP [...]
- `a2/v0` val_bpb **0.988612** (33.8 GB) — One constant: Muon MATRIX_LR 0.04 -> 0.028 (-30%) at the true baseline (HEAD_DIM 128, WARMDOWN_RATIO 0.5, DEPTH 10, 2^18, device batch 64). Nothing else. val_bpb 0.988612 vs 0.988076 baseline: a tie, [...]
- `a3/v0` val_bpb **0.989734** (33.8 GB) — Two constants: WARMDOWN_RATIO 0.5->0.7 and WINDOW_PATTERN "SSSL"->"L", so all 10 layers get the full 2048 context instead of 7 of them seeing 1024; HEAD_DIM stays 128, nothing else changed.
- `a4/v0` val_bpb **0.988490** (33.8 GB) — Added ATTN_SCALE = 0.12 to the hyperparameter block and passed it as softmax_scale to fa3.flash_attn_func, replacing FA3's default 1/sqrt(head_dim) = 0.0884 at head_dim 128 (x1.357 on attention logits [...]
- `a5/v0` val_bpb **0.988598** (33.8 GB) — EMBEDDING_LR 0.6->0.45, UNEMBEDDING_LR 0.004->0.003, MATRIX_LR 0.04->0.03, SCALAR_LR 0.5->0.375; schedule, HEAD_DIM 128, DEPTH 10, batch 2^18 all untouched.

**Outcome:** a0/v0 won; new baseline val_bpb 0.987227.

## Round 4
_Baseline entering the round: val_bpb 0.987227 (commit 77f0855)_

### Proposals
- `#148` **a0 — DEVICE_BATCH_SIZE 64 → 128 (grad_accum 2 → 1): pure throughput, zero math change**
  - why: Every axis probed so far is flat or negative and throughput has had zero runs, yet a3's all-L arm proved this box is overhead-bound (+13% attention FLOPs cost only 2% of tokens, MFU 41.9→46.4%) — so collapsing two micro-steps into one double-size fwd/bwd should buy several percent more tokens at an [...]
  - detail: One constant in the hyperparameter block of `train.py`:

- `DEVICE_BATCH_SIZE = 64` → `DEVICE_BATCH_SIZE = 128`

Nothing else. `TOTAL_BATCH_SIZE = 2**18`, `DEPTH = 10`, `ASPECT_RATIO = 64`, `HEAD_DIM = 128`, `WARMDOWN_RATIO = 0.7`, `WINDOW_PATTERN = "SSSL"`, all LRs and schedules untouched.

Mechanism. `tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN` goes 131072 → 262144, so `grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd` goes 2 → 1 (the assert `TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0` still holds exactly). Per optimizer step the inner loop runs one fwd/bwd of batch 128 instead of two of batch 64.

Why this is a *pure* throughput test, not a hyperparameter change:
- The optimizer sees the same 262144-token gradient per step, same LR schedule, same momentum schedule, same tokens per update. The only numerical difference is that the gradient is computed in one reduction instead of summed over two — an accumulation-order change, nothing more. Any val_bpb delta beyond noise is attributable to the extra tokens the run fits into 300s.
- Where the time comes from: one fewer fwd/bwd boundary per step (kernel launches + the Python dataloader `next()` + the `loss/grad_accum [...]
- `#147` **a1 — DEVICE_BATCH_SIZE 64 → 128 (grad_accum 2 → 1): free throughput**
  - why: Mathematically identical training (same 2^18-token optimizer step) but one large fwd/bwd instead of two half-sized ones, and this cell has shown the GPU is under-fed at device batch 64 — more tokens in the same 300s with zero change to the optimization.
  - detail: One constant in the hyperparameter block of `train.py`:

- `DEVICE_BATCH_SIZE = 64` → `DEVICE_BATCH_SIZE = 128`

Nothing else. `TOTAL_BATCH_SIZE = 2**18`, `DEPTH = 10`, `ASPECT_RATIO = 64`, `HEAD_DIM = 128`, `WINDOW_PATTERN = "SSSL"`, `WARMDOWN_RATIO = 0.7` and every LR stay as they are.

Mechanism. `tokens_per_fwdbwd = 128*2048 = 262144 = TOTAL_BATCH_SIZE`, so `grad_accum_steps` goes 2 → 1 (the assert still holds exactly). The optimizer sees the *same* gradient: two micro-batches of 64 scaled by 1/2 and summed, versus one batch of 128 averaged by cross-entropy — identical in expectation and identical in exact arithmetic up to reduction order. This is not a new hyperparameter; it is the same experiment with better kernel shapes. Steps, LR schedule, warmdown, weight decay and the momentum ramp are untouched (`progress` is wall-clock based).

Why I expect a real gain rather than a wash:
1. **This regime is throughput-limited, not optimization-limited.** The batch axis is flat 2^19→2^17 (#15/#17/#32), every LR knob is flat within noise (#139, #142), and every shape change failed by *losing tokens* (#49–#55). Tokens are the one quantity that has consistently ordered results. Crudely, d [...]
- `#146` **a2 — Shorten the sliding window: S = seq_len/4 (512) instead of /2**
  - why: The only non-LR effect measured outside noise this cell says less attention context is better here (all-L cost +0.0025 vs SSSL against a matched control), and nobody has sampled the shorter side — which is also strictly cheaper in FLOPs, so it buys tokens too.
  - detail: One line, inside `GPT._compute_window_sizes`:

`short_window = long_window // 2` → `short_window = long_window // 4`

Nothing else. `WINDOW_PATTERN` stays "SSSL", `HEAD_DIM` 128, `DEPTH` 10, `ASPECT_RATIO` 64, `TOTAL_BATCH_SIZE` 2**18, `DEVICE_BATCH_SIZE` 64, `WARMDOWN_RATIO` 0.7, all LRs untouched. Clean single-variable delta against the 0.987227 baseline; no partner run needed.

Mechanism. `_compute_window_sizes` gives `(short_window, 0)` to each S layer, `(long_window, 0)` to each L layer, with the last layer forced long. At depth 10, "SSSL" → S,S,S,L,S,S,S,L,S,L: 7 short layers, 3 full. Those 7 go from 1024 tokens of context to 512. FA3 just receives `window_size=(512, 0)` — same code path, no new kernels, no params, no VRAM change, no compile change.

FLOPs. The `estimate_flops` attention term is `12*h*q*effective_seq` summed over layers: `12*640*(7*1024+3*2048) = 1.02e8` now vs `12*640*(7*512+3*2048) = 7.47e7` after, so FLOPs/token 4.29e8 → 4.02e8 (0.94×). a3 (#127) showed this term badly over-predicts wall-clock (a +12.8% FLOP change cost only 2.0% of tokens), so expect a small token gain, ~2–4%, and a slightly *lower* reported MFU — expected, not a regression.

Why the dire [...]
- `#149` **a3 — Shorter sliding window: S = 512 instead of 1024**
  - why: The only measured non-LR gradient in the cell says less attention context helps (a3's all-L arm lost +0.0025 against a0's matched SSSL control), nobody has sampled the short side, and it also hands back ~6% of FLOPs as extra tokens.
  - detail: One character, in `GPT._compute_window_sizes`:

`short_window = long_window // 2` → `short_window = long_window // 4`

Nothing else. `WINDOW_PATTERN` stays "SSSL", `HEAD_DIM` 128, `DEPTH` 10, `ASPECT_RATIO` 64, `TOTAL_BATCH_SIZE` 2**18, `DEVICE_BATCH_SIZE` 64, `WARMDOWN_RATIO` 0.7, all LRs untouched. Single-variable delta against the 0.987227 baseline; no control run needed.

Mechanism. With n_layer=10 and "SSSL", windows are S,S,S,L,S,S,S,L,S,S with the last forced L, i.e. **layers 3, 7, 9 keep full 2048 context and 7 layers get the short window**. Short goes 1024 → 512. Three layers still see the whole sequence, so global context is preserved; only the local layers get more local.

Costs, zero or favourable:
- Params unchanged (window is a kernel arg). VRAM unchanged or lower.
- FLOPs: `estimate_flops` sums `12*h*q*window = 12*640*window`. Now 3*2048+7*1024 = 13312 → attn 1.022e8 of 4.29e8. After: 3*2048+7*512 = 9728 → attn 7.47e7, total ≈ **4.02e8, −6.3%**. a3 (#127) measured FLOPs→tokens as sublinear here (+12.8% FLOPs cost only 2.0% of tokens), so expect +1–3% tokens, ~1140–1160 steps. Bonus, not the bet.

Why this and not another LR knob. Round 3 showed the optimization block [...]
- `#150` **a4 — Shorter sliding window: short = 2048//4 = 512**
  - why: The window axis is the only non-LR knob with a measured ~3σ gradient (all-L cost +0.0025 vs its matched control), nobody has walked it in the good direction, and going shorter is free in params and cheaper in FLOPs — representational gain plus ~5% more tokens for one character.
  - detail: One character in `GPT._compute_window_sizes`:

- `short_window = long_window // 2` → `short_window = long_window // 4`

i.e. "S" layers attend 512 tokens instead of 1024. Nothing else: `WINDOW_PATTERN` stays `"SSSL"`, `window_sizes[-1]` is still forced to the full 2048, `HEAD_DIM` 128, `DEPTH` 10, `ASPECT_RATIO` 64, `TOTAL_BATCH_SIZE` 2**18, `DEVICE_BATCH_SIZE` 64, `WARMDOWN_RATIO` 0.7, all LRs untouched. Clean single-variable delta against the 0.987227 baseline; no control run needed.

Windows at depth 10 (i%4, last forced L): layers 0,1,2,4,5,6,8 → 512; layers 3,7,9 → 2048.

Cost, from `estimate_flops`: `attn_flops = Σ 12*h*q*window = 12*640*(7*512+3*2048) = 7.47e7`, down from `12*640*(7*1024+3*2048) = 1.02e8`. Total FLOPs/token 4.01e8 vs 4.29e8, **−6.5%**. Zero change in params, optimizer, VRAM or line count. Expect ~5% more tokens (a3 measured weak FLOP→token coupling: +12.8% FLOPs cost only 2.0% of tokens, MFU 41.94→46.38%), so ~305M tokens at ~1170 steps.

Why the direction:
1. **Measured.** a3's all-L arm (#127) and a0's (#124) differ in exactly one character; the window term is 0.989734 − 0.987227 = **+0.00251**, ~3σ on the round-3 arm-to-arm sd of 8e-4, and a3's arm had th [...]
- `#151` **a5 — RoPE base 10000 → 1000 (match rotary spectrum to the 2048-token context)**
  - why: At seq_len 2048 with head_dim 128 roughly half the rotary frequency pairs have periods far longer than the context (slowest ≈ 2π·10000 = 62832 positions) and are near-constant dead channels; shrinking the base repacks the spectrum into the range the model actually sees, at exactly zero cost in param [...]
  - detail: One value, one line, in `GPT._precompute_rotary_embeddings`:

`def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):` → `base=1000`.

No new constant (a module-level `ROPE_BASE` would have to sit above the class, since defaults evaluate at class-definition time — not worth the churn for one number). Both call sites (`__init__` and `init_weights`, which recomputes cos/sin after `to_empty`) pass no `base`, so both pick it up. Nothing else changes: HEAD_DIM 128, DEPTH 10, ASPECT_RATIO 64, WINDOW_PATTERN "SSSL", 2^18, device batch 64, WARMDOWN_RATIO 0.7, all LRs.

Mechanism. `inv_freq = base**(-c/head_dim)` for c = 0,2,…,126, so periods run from 2π to 2π·base. At base 10000 the slowest ~30 of 64 pairs have periods > 2048: over the whole training context they rotate less than a radian, so cos≈1, sin≈0 and those channels carry almost no positional signal while still occupying half the q/k width (QK-RMSNorm then normalises over that mix). At base 1000 the slowest period is 6283 — still longer than any span — but every pair now discriminates positions inside 2048, and inside the 1024 windows on 7 of 10 layers.

Why this axis is live here rather than another f [...]

### Messages
- `#152` **a2**: a2 SLOT CLAIM, posted BEFORE launching. Three of us proposed the IDENTICAL short-window change: #146 (me), #149 a3, #150 a4 — same one character, short_window = long_window // 2 -> // 4. Under the tie-break rule the group has used all cell (#81, lowest index runs the contested square) **a2 owns the short-window probe and I am launching it now.** a3, a4: treat it as taken, do NOT run a second copy — and please do not both flee to the same replacement square, which is exactly how round 1 burned four slots.

Also on the table: a0 (#148) and a1 (#147) proposed the IDENTICAL DEVICE_BATCH_SIZE 64 -> 128. Lowest index gives that to a0; a1 should defect. a5 (#151) has RoPE base 1000 alone — nobody duplicate it.

Suggested clean defections for a1/a3/a4, one each, so six slots buy six bits: (i) max-autotune-no-cudagraphs, which I pre-claimed in #104 and am now RELEASING since I am running the wind [...]
- `#153` **a5**: a5 SLOT CLAIM (#151), posted BEFORE launching. Tally: a0 (#148) and a1 (#147) are the identical DEVICE_BATCH_SIZE 64->128; a2 (#146), a3 (#149) and a4 (#150) are the identical short_window //2 -> //4. My RoPE-base arm is the only unclaimed square, so I am running it exactly as proposed: one value, _precompute_rotary_embeddings base=10000 -> 1000, nothing else (HEAD_DIM 128, DEPTH 10, SSSL, 2^18, device batch 64, WARMDOWN 0.7, all LRs).

Under the lowest-index rule a0 owns device-batch-128 and a2 owns short-window-512. a1/a3/a4: three copies of the window change is round 0 again. If one of you wants a second point rather than a replicate, the useful ones are short_window = long//8 (256) to get the SHAPE of the window axis, or WINDOW_PATTERN 'SSSS' at the current 1024 (fewer full-context layers, orthogonal to window length). Say which before you launch.

One caution on the window cluster's [...]
- `#154` **a0**: a0 SLOT CLAIM, posted BEFORE launching. Proposal tally this round: DEVICE_BATCH_SIZE 128 x2 (#148 me, #147 a1); short_window //4 x3 (#146 a2, #149 a3, #150 a4); RoPE base x1 (#151 a5). Under the lowest-index rule (#81, #118) I own device batch 128 and I am launching it now — edit is in (train.py:451, DEVICE_BATCH_SIZE 64 -> 128, nothing else, grad_accum 2 -> 1).

a1: device batch 128 is COVERED. a2/a3/a4: three copies of short_window //4 is the round-0 failure mode again; under the same rule a2 owns it and a3/a4 should defect — and NOT to the same square. Two obvious unclaimed defections: (a) WINDOW_PATTERN 'SSSL' -> 'SSSS' (only the forced-last layer stays long), which is the OTHER end of the same window axis and independent of the short-length change; (b) max-autotune compile, still zero runs in 23 (#104). Say which in one line before launching.

Predictions on the record so I cannot r [...]
- `#155` **a4**: a4 SLOT CLAIM, posted BEFORE launching. Three of six proposals are the IDENTICAL short_window //4 (#146 a2, #149 a3, #150 me) and two are the identical DEVICE_BATCH_SIZE 128 (#148 a0, #147 a1). Under the lowest-index rule a2 owns //4 and a0 owns device-batch-128. I am NOT running a second copy of my own proposal.

I am running the fallback I pre-registered in #150's notes, verbatim: 'If nobody claims SSSS and someone claims //4, I take SSSS instead.' Nobody claimed it, a2/a3 claimed //4, so:

  WINDOW_PATTERN = 'SSSL' -> 'SSSS'

One string, nothing else. HEAD_DIM 128, DEPTH 10, ASPECT_RATIO 64, 2^18, device batch 64, WARMDOWN_RATIO 0.7, short_window stays long//2 = 1024, all LRs untouched.

Why this is the right second point and not a near-duplicate: the window axis is two-dimensional and a2/a3 and I are each moving a different coordinate. They shrink the SHORT layers (1024 -> 512, 7 lay [...]
- `#156` **a3**: a3 SLOT CLAIM, posted BEFORE launching. Round-4 tally: THREE of us proposed the identical short-window change (#146 a2, #149 me, #150 a4) and TWO proposed the identical DEVICE_BATCH_SIZE 128 (#148 a0, #147 a1). a5 (#151) is alone on RoPE base. Under the lowest-index rule (#81): a2 owns S=512, a0 owns device batch 128, a5 owns RoPE. a1, a4 and I must land on THREE DIFFERENT squares - say which in one line before you launch.

I am NOT running S=512. I am running the NEXT point on the same axis: short_window = long_window // 8, i.e. S = 256 (one character in _compute_window_sizes, nothing else; WINDOW_PATTERN stays SSSL so layers 3,7,9 keep the full 2048).

Why: the window axis is the only non-LR axis with a measured gradient (a0's SSSL 0.987227 vs my all-L 0.989734, byte-identical apart from one word => +0.00251 for MORE context, ~3 sigma), and it is measured at ONE point. With a2 at 512 a [...]
- `#157` **a1**: a1 SLOT CLAIM, posted BEFORE launching. Tally: a0 (#148) and I (#147) proposed the IDENTICAL DEVICE_BATCH_SIZE 64->128; a2 (#146), a3 (#149) and a4 (#150) proposed the IDENTICAL short_window //2 -> //4. So two squares have 2x and 3x coverage and a5 (#151) is alone on RoPE base.

Under the lowest-index rule (#81) a0 owns pure db128 and a2 owns pure short-window. I am NOT running a fourth/third copy of either.

I am running BOTH TOGETHER: DEVICE_BATCH_SIZE 64->128 (grad_accum 2->1) AND short_window = long_window//4 (S layers attend 512 instead of 1024). Two constants, nothing else; HEAD_DIM 128, DEPTH 10, WINDOW_PATTERN SSSL, WARMDOWN 0.7, all LRs untouched.

Why this is the right use of my slot rather than a defection to an unclaimed knob:

1. BOTH terms get measured independently this round by other agents. a0 is my control for the window term; a2/a3/a4 are my control for the device-batc [...]
- `#158` **a2** _(re: #152)_: a2 RESULT (#152, short window S=512): **val_bpb 0.982712 vs the 0.987227 baseline, -0.0045.** That is ~5.5 sigma on the round-3 arm-to-arm sd of 8e-4, and -0.0058 against the round-3 pool mean (0.9885), which is the fairer control given a0's point in #141 that the winning baseline is an order statistic. Largest effect measured at fixed model size in this cell.

Numbers: 1152 steps, 302.0M tokens (baseline 293.9M, +2.8%), MFU 40.44% (baseline 42.03% - down, exactly as predicted, since FLOPs/token fell 4.29e8 -> 4.02e8 at roughly constant wall-clock), peak VRAM 34.6GB, 85.9M params.

**The window axis is monotone and has not turned over.** Three points, all at warmdown 0.7 / HD128, differing in nothing else:
  S=512  -> 0.982712
  S=1024 -> 0.987227 (a0, #124)
  all-L  -> 0.989734 (a3, #127)
Less context is better, twice in a row, with no sign of a floor. **The next probe is short_window = [...]
- `#159` **a5** _(re: #151)_: a5 RESULT (#151, RoPE base 10000 -> 1000): NEGATIVE. val_bpb 0.991196 vs the 0.987227 baseline, +0.0040, about 5 sigma on the round-3 arm-to-arm sd of 8e-4.

It is the cleanest matched pair anyone has produced: 1120 steps / 293.6M tokens / MFU 42.01% / 34.6GB against the baseline's 1121 / 293.9M / 42.03% / 34.6GB. Only the values inside two precomputed buffers differed, so there is no throughput, shape or eval confound. Mean-last-60 train loss (a3's lower-variance metric from #138) agrees: 2.7744 vs the baseline's 2.7625. The model genuinely trained worse.

I was arithmetically right and inferentially wrong. At base 10000 with head_dim 128 roughly half the frequency pairs really do have periods longer than 2048 and rotate less than a radian over the whole context. I called them dead channels. They are not dead - they are doing work, and compressing the spectrum to 'fit' the context destr [...]
- `#160` **a0**: a0 RESULT (#154, DEVICE_BATCH_SIZE 64 -> 128): val_bpb 0.985624 vs the 0.987227 baseline, -0.0016. I am NOT claiming that as a win, and the reason is that my own predicted mechanism did not appear.

Numbers: 1135 steps / 297.5M tokens / MFU 42.56% / peak VRAM 67.8GB / 300.1s, against the baseline's 1121 / 293.9M / 42.03% / 34.6GB.

1. THE THROUGHPUT BET LARGELY FAILED. I predicted 3-8% more tokens from collapsing grad_accum 2 -> 1. I got +1.2%. The gradient-accumulation boundary at 2^18 / depth 10 is worth about 1% of wall-clock, full stop. By this cell's token/loss calibration (~0.03-0.04 bpb per doubling of tokens) 1.2% buys roughly -0.0006, so AT MOST a third of my -0.0016 is the change and the rest is a ~2 sigma draw on a surface whose arm-to-arm sd is 8e-4. Averaged train loss says the same thing and says it against me: mean of the last 60 logged values is 2.7646 vs the baseline's 2 [...]
- `#161` **a4** _(re: #155)_: a4 RESULT (#155, WINDOW_PATTERN 'SSSL' -> 'SSSS'): TIE. val_bpb 0.986958 vs the 0.987227 baseline, -0.00027 = 0.3 sigma. Not a win, and I will not call it one.

Numbers: 1124 steps, 294.6M tokens, MFU 40.59%, peak VRAM 34.6GB, FLOPs/token 4.13e8 (-3.5%), 300.2s. Train loss, using the mean-of-last-60 metric a3 established in #138 rather than the single noisy EMA sample: 2.7622 against a0's matched control 2.7625. Indistinguishable on both metrics.

Two results, and the second one matters more than mine.

1. THE WINDOW AXIS IS FLAT ON THE SHORT SIDE AND SHOULD BE CLOSED. I removed two of the three global layers (3 and 7 drop 2048 -> 1024; only the forced last layer stays full) and nothing happened. Put together with a3's r03 arm:

  all-L, every layer 2048      -> 0.989734  (+0.0025 vs matched control)
  SSSL, 3 global + 7 at 1024   -> 0.987227  (baseline)
  SSSS, 1 global + 9 at 1024   -> [...]
- `#162` **a1** _(re: #157)_: a1 RESULT (#157, short window 512 + DEVICE_BATCH_SIZE 128): **val_bpb 0.981817 vs the 0.987227 baseline, -0.0054.** That is ~7 sigma on the round-3 arm-to-arm sd of 8e-4 and the largest single improvement in this cell since depth 8->10. Commit d1719d9.

Numbers: 1168 steps, 306.2M tokens (baseline 293.9M, +4.2%), MFU 41.00% (baseline 42.03%), peak VRAM 67.8GB, FLOPs/token 4.29e8 -> 4.01e8, final smoothed train loss 2.752 / mean-last-60 2.758, training_seconds 300.1.

**The mechanism split is the important part, and it does NOT favour the half I proposed.**

tok/s went 0.979M -> 1.020M, +4.2%. But FLOPs/token fell 6.5%, so the actual FLOP rate DROPPED 2.5% (that is the MFU line). In other words every extra token came from the cheaper attention; collapsing grad_accum 2->1 delivered nothing measurable in throughput. a0: your pure db128 arm is the clean control and I expect it to land near b [...]
- `#163` **a3** _(re: #156)_: a3 RESULT (#156, short window S = 2048//8 = 256): val_bpb 0.984406 vs the 0.987227 baseline, -0.00282. That is ~3.5 sigma on the round-3 arm-to-arm sd of 8e-4, and -0.0041 against the round-3 pool mean (0.9885), which is the fairer bar since the baseline is an order statistic (a0, #141). Verified real: arena_run.json says round 4, agent a3, exit 0.

Numbers: 1174 steps, 307.8M tokens (baseline 293.9M, +4.7%), peak VRAM 34.6GB, 85.9M params, 300.2s. MFU reads 39.79% vs 42.03% but that is an artifact - estimate_flops drops the attention term, so the denominator shrank; raw throughput ROSE, 1.02M tok/s vs 973K. Mean-of-last-60 train loss 2.761 vs a0's 2.7625 on the same measure, i.e. val and train agree.

**The window axis now has four points at otherwise matched settings, and it is monotone:**
  2048 (all-L, my r3 arm)  0.989734
  1024 (baseline)          0.987227
   512 (a2/a4 this round) [...]
- `#171` **a5** _(re: #163)_: a5 -> a3 (#163): your four points are not monotone — they turn over, and your own arm is the turning point. 2048: 0.989734 / 1024: 0.987227 / 512: 0.982712 / 256: 0.984406. S=256 is +0.0017 WORSE than S=512, and it saw MORE tokens (307.8M vs a2's 302.0M). So 512 is an interior minimum, bracketed on both sides by measured points.

That makes your run the most informative one of the round even though it lost: it is the arm that rules out the throughput explanation of the whole window cluster. Within the window family the ranking is anti-correlated with tokens — you had the most tokens and finished third of four. The win is representational.

Fitting a parabola in log2(window) to your three points puts the vertex at ~440 and predicts less than 2e-4 below S=512. That is a quarter of the noise floor. Nobody should spend one of the last 7 runs on 384 or 768.
- `#172` **a1** _(re: #163)_: a1 -> a3 (#163): 'the window axis is monotone' is the one claim from this round I think is wrong, and your own run is what breaks it. I diffed a2's train.py against yours: they differ in EXACTLY one character, '// 4' vs '// 8' on line 199. Same db64, same warmdown 0.7, same HD128, same everything. So the pair is clean and it reads:

  S=512 (a2) -> 0.982712
  S=256 (you) -> 0.984406

256 is +0.0017 WORSE than 512, about 2 sigma on the 8e-4 arm-to-arm sd, and your arm also saw MORE tokens (307.8M vs 302.0M) so the token channel argues the wrong way too. The four points are 0.989734 / 0.987227 / 0.982712 / 0.984406 - that is a U with its minimum at 512, not a monotone descent. You bracketed the axis in one run, which is a better result than another point on a slope, but it is the opposite of what your message says. Nobody should run S=128.
- `#173` **a0** _(re: #163)_: a0 -> a3 (#163): your four points are NOT monotone — your own run is the turnover, and the group is one slot away from acting on the wrong shape.

  2048 (your r03 all-L)  0.989734
  1024 (my r03 arm)      0.987227
   512 (a2)              0.982712
   256 (you)             0.984406

256 is +0.0017 WORSE than 512, ~2 sigma the wrong way. The axis is U-shaped with the minimum at or near 512, not a slope that keeps paying. And the cleanest part of your own data is what kills the 'it is just tokens' reading: you got MORE tokens than a2 (307.8M vs 302.0M) and scored worse. So the 1024 -> 512 win is representational, and it is spent — nobody should run 128, and S=384/768 are worth at most ~0.001 from here. Your arm bracketed the optimum in one run, which is exactly what you said in #156 you were buying; it just came out the other way from your headline.
- `#174` **a4** _(re: #158)_: a4 -> a2 (#158): 'the window axis is monotone and has not turned over; the next probe is a shorter window' is refuted by a3's arm, which landed after you wrote. S=256 scored 0.984406 against your S=512's 0.982712 — it is WORSE by 0.0017, and the token correction makes the turnover bigger, not smaller: a3 ran 307.8M tokens to your 302.0M, so at matched tokens the gap is ~0.0026.

Please do not run 128, and I am asking a3 not to either. Fit a parabola in log2(w) through the three matched points (1024/512/256 = 0.987227/0.982712/0.984406): the vertex sits at w = 437 and is worth 0.00016 over 512. That is a fifth of the noise floor. The window axis is closed at 512 and any further probe on it — 384, 448, 128 — is a slot spent buying a number we can already read off the curve, exactly the trap we fell into with pure-2^18 in round 0.

Your run is still the round's most valuable measurement and [...]
- `#175` **a0**: **The window axis turned over this round, and the thing that closed it is a3's own run.**

All at db64 / HD128 / warmdown 0.7 / SSSL, one character apart:

| S | val_bpb | tokens |
|---|---|---|
| 2048 (a3 r03) | 0.989734 | 287.3M |
| 1024 (my r03 arm) | 0.987227 | 293.9M |
| **512 (a2)** | **0.982712** | 302.0M |
| 256 (a3) | 0.984406 | 307.8M |

**256 is +0.0017 worse than 512 while seeing 2% MORE tokens.** That kills two claims: a3's "monotone" (#163) — it is U-shaped, minimum ~512 — and the idea that the window win was a throughput win. More tokens, worse loss. The gain is representational and now spent. **Nobody run S=128/256; S=384/768 are worth ≤0.001 from a bracketed optimum.**

**Device batch 128 is real but small, and already banked.** I diffed a1's and a2's `train.py`: they differ in `DEVICE_BATCH_SIZE` and one comment word, nothing else. Two independent matched pairs:
- at S= [...]
- `#176` **a1**: **The window axis turned over. It is bracketed, not monotone — the optimum is 512 and we should stop probing it.**

I diffed every r04 `train.py`. Three clean single-character matched pairs came out:

| pair | differs by | val_bpb | term |
|---|---|---|---|
| a0 r3 → a2 | `//2`→`//4` only | 0.987227 → 0.982712 | **−0.0045** |
| a0 r4 → a1 | `//2`→`//4` only | 0.985624 → 0.981817 | **−0.0038** |
| a2 → a3 | `//4`→`//8` only | 0.982712 → 0.984406 | **+0.0017** |

**1. S=256 is worse than S=512.** a2's and a3's files differ in exactly one character (line 199), same db64, warmdown, HD128. Four points: 0.989734 (2048) / 0.987227 (1024) / **0.982712 (512)** / 0.984406 (256) — a U, not a slope. a3 (#163) called it monotone; it isn't, and a3's arm saw *more* tokens (307.8M vs 302.0M), so the token channel argues against that too. **Nobody run S=128 or 384** — the 512↔256 gap is 2σ, so a midpoint [...]
- `#177` **a5**: 
- `#178` **a2**: 
- `#179` **a3**: **The window axis turned over at 512, and the winner's second half is noise that cost 33GB of VRAM.**

**1. My arm brackets the axis. It is closed — do not spend a slot on it.** Four points, otherwise matched (warmdown 0.7, HD128, SSSL, db64):

| S | val_bpb | tokens |
|---|---|---|
| 2048 (all-L, my r3 arm) | 0.989734 | 287.3M |
| 1024 (r3 baseline) | 0.987227 | 293.9M |
| **512 (a2)** | **0.982712** | 302.0M |
| 256 (me) | 0.984406 | 307.8M |

512 is the minimum. My 256 arm is +0.0017 worse than 512 *while seeing 1.9% more tokens*, so the representational cost is slightly larger than the raw delta. Add a4's SSSS tie (−0.00027) and both coordinates are done: **length optimum 512, global-layer count flat.** Nobody probe 384, 128 or SSSS+512. I retract the "has not turned over, next probe S=128" line in my own #163 — a2's number wasn't in when I wrote it.

**2. a1's win is ~0.0045 window [...]
- `#180` **a4**: **I was wrong in #161, the window axis turned over, and it is now closed. Here is the arithmetic so nobody spends a slot on it.**

**1. Retracting my own claim.** I wrote "the window axis is flat on the short side, close it." a2 (−0.0045) and a3 (−0.0028) refute that. My arm was simply a much smaller perturbation: I halved 2 layers, they halved 7, and mine sat in the flat 1024→2048 region. Not a contradiction — an underpowered probe I over-generalised.

**2. But it HAS turned over, and nobody should probe shorter.** Matched points (wd 0.7, HD128, db64):

| short w | val_bpb | tokens |
|---|---|---|
| 2048 (all-L) | 0.989734 | 287.3M |
| 1024 | 0.987227 | 293.9M |
| **512** | **0.982712** | 302.0M |
| 256 | 0.984406 | 307.8M |

256 is worse than 512 *despite 1.9% more tokens*. A parabola in log2(w) puts the vertex at **w = 437, worth 0.00016 over 512** — a fifth of the noise floor. **Don' [...]

### Experiments run
- `a0/v0` val_bpb **0.985624** (66.2 GB) — One constant: DEVICE_BATCH_SIZE 64 -> 128, so the 2^18-token optimizer step is one fwd/bwd instead of two accumulated micro-steps; identical math, nothing else changed.
- `a1/v0` val_bpb **0.981817** (66.2 GB) — Two constants: short_window = long_window//4 (S layers attend 512 instead of 1024; 3 of 10 layers keep full 2048) and DEVICE_BATCH_SIZE 64->128 so grad_accum goes 2->1 at the same 2^18 total batch.
- `a2/v0` val_bpb **0.982712** (33.8 GB) — One character in GPT._compute_window_sizes: short_window = long_window // 2 -> // 4, so the 7 'S' layers attend 512 tokens instead of 1024 (3 layers keep the full 2048); nothing else changed.
- `a3/v0` val_bpb **0.984406** (33.8 GB) — One character in _compute_window_sizes: short_window = long_window // 2 -> // 8, so the 7 'S' layers attend 256 tokens instead of 1024; layers 3, 7, 9 keep the full 2048. Nothing else changed.
- `a4/v0` val_bpb **0.986958** (33.8 GB) — One string: WINDOW_PATTERN "SSSL" -> "SSSS", so layers 0-8 attend 1024 and only the forced last layer keeps the full 2048 context (was layers 3, 7, 9 global). short_window stays long//2; nothing else [...]
- `a5/v0` val_bpb **0.991196** (33.8 GB) — One value: _precompute_rotary_embeddings base=10000 -> 1000; nothing else changed. val_bpb 0.991196 vs 0.987227 baseline (+0.0040).

**Outcome:** a1/v0 won; new baseline val_bpb 0.981817.

