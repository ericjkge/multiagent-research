# Shared research directory (open protocol)

Baseline at start of the cell: **val_bpb 1.012531** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 36 runs. Used so far: a0: 0 left.

## Approaches (slots)

- `#2` **a0** — depth/width & LR scaling: Sweep DEPTH (model size) and learning rates/schedule (warmdown ratio, matrix_lr, embedding_lr) to find the best compute-optimal point within the fixed 5-min budget; will not assume DEPTH=8 or current LRs are optimal for this token budget.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 1 | a0 | 1.039018 | ok | `d844cc6` | DEPTH=6 (narrower/shallower, more steps in 5min) |
| 2 | a0 | 1.002905 | ok | `520bef0` | DEPTH=10 (deeper/wider than baseline 8) |
| 3 | a0 | — | crash | `875f66f` | DEPTH=14 (even deeper) |
| 4 | a0 | 1.069575 | ok | `f39b038` | DEPTH=14, DEVICE_BATCH_SIZE=64 to fit memory |
| 5 | a0 | 1.017372 | ok | `94c966f` | DEPTH=12, DEVICE_BATCH_SIZE=64 |
| 6 | a0 | 1.009238 | ok | `18a200a` | DEPTH=9 (same width as depth10, one fewer layer), batch128 |
| 7 | a0 | 1.004630 | ok | `5495017` | DEPTH=10, WARMDOWN_RATIO=0.3 (shorter cooldown) |
| 8 | a0 | 1.004438 | ok | `82b1f8d` | DEPTH=10, WARMDOWN_RATIO=0.7 (longer cooldown) |
| 9 | a0 | 1.006833 | ok | `b0109ae` | DEPTH=10, MATRIX_LR=0.03 (lower Muon LR) |
| 10 | a0 | 1.006996 | ok | `866e060` | DEPTH=10, MATRIX_LR=0.05 (higher Muon LR) |
| 11 | a0 | 0.988032 | ok | `3c032bf` | DEPTH=10, TOTAL_BATCH_SIZE=2^18 (half, more optimizer steps) |
| 12 | a0 | 0.987682 | ok | `e0ec5eb` | DEPTH=10, TOTAL_BATCH_SIZE=2^17 (quarter, DBS=64) |
| 13 | a0 | 0.998458 | ok | `3175ef3` | DEPTH=10, TOTAL_BATCH_SIZE=2^16 (DBS=32, even smaller batch) |
| 14 | a0 | 0.985612 | ok | `6f769ef` | DEPTH=10, TOTAL_BATCH_SIZE=196608 (between 2^17 and 2^18) |
| 15 | a0 | 0.985150 | ok | `f3d5ccb` | DEPTH=10, TOTAL_BATCH_SIZE=163840 (80*2048) |
| 16 | a0 | 0.986190 | ok | `4059684` | DEPTH=10, TOTAL_BATCH_SIZE=147456 (72*2048) |
| 17 | a0 | 0.987218 | ok | `962c102` | DEPTH=10, TOTAL_BATCH_SIZE=180224 (88*2048) |
| 18 | a0 | 0.989457 | ok | `6d185e4` | DEPTH=12 with new best TBS=163840 |
| 19 | a0 | 0.987137 | ok | `a1ee048` | DEPTH=11 with TBS=163840 |
| 20 | a0 | 0.985593 | ok | `c8c22a9` | DEPTH=10 TBS=163840, MATRIX_LR=0.035 |
| 21 | a0 | 0.986404 | ok | `04e8681` | DEPTH=10 TBS=163840, EMBEDDING_LR=0.4 |
| 22 | a0 | 0.986376 | ok | `b4dead0` | DEPTH=10 TBS=163840, EMBEDDING_LR=0.8 |
| 23 | a0 | 0.984334 | ok | `60cb442` | DEPTH=10 TBS=163840, WARMUP_RATIO=0.05 |
| 24 | a0 | 0.986681 | ok | `c17ad41` | DEPTH=10 TBS=163840, WARMUP_RATIO=0.1 |
| 25 | a0 | 0.983990 | ok | `56113cf` | DEPTH=10 TBS=163840, WARMUP_RATIO=0.03 |
| 26 | a0 | 0.984872 | ok | `5e9be82` | DEPTH=10 TBS=163840, WARMUP_RATIO=0.02 |
| 27 | a0 | 0.981159 | ok | `d12fa60` | DEPTH=10 TBS=163840 WARMUP=0.03, WEIGHT_DECAY=0.1 |
| 28 | a0 | 0.983578 | ok | `b9c054f` | DEPTH=10 TBS=163840 WARMUP=0.03, WEIGHT_DECAY=0.0 |
| 29 | a0 | 0.981856 | ok | `3b64426` | DEPTH=10 TBS=163840 WARMUP=0.03, WEIGHT_DECAY=0.15 |
| 30 | a0 | 0.979324 | ok | `8f497ee` | DEPTH=9 reconfirm at WD=0.1/warmup=0.03/TBS=163840 |
| 31 | a0 | 0.986418 | ok | `388261f` | DEPTH=8 (orig baseline depth) with new TBS/warmup/WD tuning |
| 32 | a0 | 0.980041 | ok | `87611d4` | DEPTH=9, WEIGHT_DECAY=0.05 |
| 33 | a0 | 0.979559 | ok | `5724ab8` | DEPTH=9, TOTAL_BATCH_SIZE=131072 (smaller batch) |
| 34 | a0 | 0.981430 | ok | `b265e6a` | DEPTH=9 best config, ADAM_BETAS=(0.85,0.95) |
| 35 | a0 | 0.979623 | ok | `a5833af` | Reconfirm best config: DEPTH=9, TBS=163840, WARMUP=0.03, WD=0.1 |
| 36 | a0 | 0.980533 | ok | `716a032` | DEPTH=9 best config, WARMDOWN_RATIO=0.4 (last run, testing untested lever at D9) |

**Best so far: val_bpb 0.979324 by a0 at commit `8f497ee`.**

## Findings (append-only broadcast)

- `#9` **a0** [commit `520bef0`]: Depth sweep at fixed 5-min budget (batch128 unless OOM): D6=1.039018, D9=1.009238, D10=1.002905 (best), D12=1.017372(bs64), D14=1.069575(bs64, OOM at bs128). D10 (model_dim=640) is a clear local optimum vs baseline D8=1.012531. Deeper models don't get enough steps in 5min to pay off.
- `#15` **a0** [commit `3c032bf`]: Big win: DEPTH=10 + TOTAL_BATCH_SIZE=2^18 (halved from 2^19, more optimizer steps in the 5min budget) -> val_bpb=0.988032, vs DEPTH=10+TBS=2^19 -> 1.002905, vs baseline D8+TBS=2^19 -> 1.012531. Smaller effective batch (more, noisier Muon/Adam steps) beats larger batch under fixed wallclock budget.
- `#19` **a0** [commit `6f769ef`]: Refining batch size sweep at DEPTH=10: TBS=2^19->1.002905, 2^18->0.988032, 2^17->0.987682, 2^16->0.998458, 196608(96*2048)->0.985612 (new best). Sweet spot appears to be TBS around 150-200K tokens, not extreme small or the original 512K.
- `#25` **a0** [commit `f3d5ccb`]: Confirmed DEPTH=10 remains best depth even after re-tuning batch size: with TBS=163840, D10=0.985150 (best), D11=0.987137, D12=0.989457. Locking DEPTH=10, TOTAL_BATCH_SIZE=163840 (80*2048) as current best config.
- `#33` **a0** [commit `56113cf`]: Adding small warmup helps at DEPTH=10/TBS=163840: WARMUP_RATIO 0.0->0.985150, 0.02->0.984872, 0.03->0.983990 (new best), 0.05->0.984334, 0.1->0.986681. Best around 0.03 (~9s warmup). Small effect (~0.001) but consistent direction; likely because many more (noisier) optimizer steps per second at low batch benefit from brief LR ramp-up.
- `#37` **a0** [commit `d12fa60`]: Weight decay sweep at DEPTH=10/TBS=163840/WARMUP=0.03: WD=0.2(orig)->0.983990, WD=0.1->0.981159 (new best), WD=0.0->0.983578, WD=0.15->0.981856. Lower WD (~0.1) is better than the original 0.2 default under this smaller-batch/more-steps regime.
- `#40` **a0** [commit `8f497ee`]: With WD=0.1, WARMUP_RATIO=0.03, TBS=163840 tuned at D10, re-sweeping depth shows D9=0.979324 (new best, beats D10=0.981159) while D8=0.986418 is worse. Depth optimum shifted down once WD/warmup were retuned -- hyperparameter interactions matter, single-variable sweeps can mislead.
- `#45` **a0** [commit `8f497ee`]: FINAL best result: val_bpb=0.979324 (reconfirmed at 0.979623, commit a5833af) with DEPTH=9, TOTAL_BATCH_SIZE=163840 (80*2048), WARMUP_RATIO=0.03, WEIGHT_DECAY=0.1 (all else default). Down from baseline 1.012531, a ~3.3% bpb reduction. Key levers in order of impact: (1) TOTAL_BATCH_SIZE reduced from 2^19 to ~163840 (more, smaller optimizer steps in fixed wallclock -- biggest single win), (2) WEIGHT_DECAY 0.2->0.1, (3) small WARMUP_RATIO~0.03, (4) DEPTH re-tuned to 9 after the above changes shifted the optimum down from 10. ADAM_BETAS and MATRIX_LR/EMBEDDING_LR perturbations around the defaults were all neutral-to-negative (within ~0.001-0.002 noise).
- `#49` **a0** [commit `716a032`]: Final check: at DEPTH=9 best config, WARMDOWN_RATIO=0.4 -> 0.980533, slightly worse than the default 0.5 (0.979324/0.979623 reconfirmed). Default warmdown holds up at D9 too. This exhausts the cell's 36 runs; final best remains commit 8f497ee, val_bpb=0.979324 (DEPTH=9, TOTAL_BATCH_SIZE=163840, WARMUP_RATIO=0.03, WEIGHT_DECAY=0.1, all else default).

## Disconfirmations (negative results, attempts to falsify)

_No disconfirmations yet._

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

