# Shared research directory (open protocol)

_You are `a2`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997333** (commit 228791f)

**Training runs remaining in this cell: 4 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 4 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#6` **a2** — depth/width scaling + batch size tradeoff: I will sweep model depth (and derived width) and device batch size / grad-accum under the fixed 5-min budget to find a better compute-optimal point than the DEPTH=8 baseline; I will not assume the current LR schedule is optimal for a different depth, so I'll rescale LR via the existing 1/sqrt(dmodel) rule already in the code.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 3 | a2 | 1.001200 | ok | `e2c05e6` | DEPTH=10 (wider/deeper model, same aspect ratio) |
| 7 | a2 | 1.043780 | ok | `d69a08f` | DEPTH=6 (narrower/shallower, more steps in budget) |
| 12 | a2 | 0.988582 | ok | `1cd5fad` | DEPTH=8 baseline but TOTAL_BATCH_SIZE halved to 2^18 (more, noisier updates) |
| 16 | a2 | 0.994878 | ok | `ed699f6` | DEPTH=8, TOTAL_BATCH_SIZE=2^17, DEVICE_BATCH_SIZE=64 (even more/noisier updates) |
| 21 | a2 | 0.991420 | ok | `015614d` | Best batch (2^18) + shorter warmdown ratio 0.3 (more steps at full LR) |
| 26 | a2 | 0.988312 | ok | `896a599` | Best batch (2^18) + higher EMBEDDING_LR 0.6->0.8 (final run in my share) |

**Best so far: val_bpb 0.988312 by a2 at commit `896a599`.**

## Findings (append-only broadcast)

- `#50` **a2** [commit `1cd5fad`]: DEPTH=8 baseline with TOTAL_BATCH_SIZE halved from 2^19 to 2^18 (grad_accum 2->1 at DEVICE_BATCH_SIZE=128) gives val_bpb 0.988582, beating baseline 0.997333. More frequent, noisier optimizer steps within the fixed 5-min budget outweigh the loss of gradient-averaging quality. This is my best so far.
- `#86` **a2** [commit `896a599`]: Final result for a2: best val_bpb 0.988312 at commit 896a599 (DEPTH=8, TOTAL_BATCH_SIZE=2^18 i.e. halved from baseline's 2^19, WARMDOWN_RATIO=0.5, EMBEDDING_LR=0.8). This beats the cell baseline 0.997333 by ~0.009. The embedding LR bump (0.6->0.8) over the 2^18-batch config (0.988582) made no real difference (0.988312, within noise). Summary of my depth/batch sweep: DEPTH=6 and DEPTH=10 both hurt (1.04, 1.00) -- baseline DEPTH=8 is near depth-optimal for the 5-min budget. The real lever was halving TOTAL_BATCH_SIZE from 2^19 to 2^18 (more, noisier optimizer steps in fixed wall-clock time) which gave the biggest single gain; going further to 2^17 overshot and got worse (0.9949) than 2^18 but still beat baseline.

## Disconfirmations (negative results, attempts to falsify)

- `#15` **a2** [commit `e2c05e6`]: DEPTH=10 (vs baseline DEPTH=8) gives val_bpb 1.001200, worse than baseline 0.997333. Under the fixed 5-min budget, the extra capacity from going deeper/wider does not compensate for fewer optimizer steps -- baseline depth looks close to compute-optimal for this budget, if not slightly too deep already.
- `#61` **a2** [commit `ed699f6`]: Pushing further to TOTAL_BATCH_SIZE=2^17 (DEVICE_BATCH_SIZE=64) gives val_bpb 0.994878 -- still better than baseline 0.997333 but worse than TOTAL_BATCH_SIZE=2^18 (0.988582). So halving batch size once (2^19->2^18) helps, but halving again overshoots into too-noisy territory. 2^18 is a local sweet spot, not 'smaller is always better'.
- `#73` **a2** [commit `015614d`]: [weak, single run] At the winning TOTAL_BATCH_SIZE=2^18, shortening WARMDOWN_RATIO from 0.5 to 0.3 gives val_bpb 0.991420, worse than 0.988582 with WARMDOWN_RATIO=0.5. The default warmdown schedule pairs well with the smaller-batch config; no need to retune it.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

