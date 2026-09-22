# Shared research directory (open protocol)

_You are `a5`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 0.997359** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#8` **a5** — depth/width and batch-size scaling: Explore whether DEPTH=8 and TOTAL_BATCH_SIZE=2^19 are compute-optimal for the 5-min budget; will not assume current LR schedule or window pattern is optimal either, but will vary one axis at a time.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 5 | a5 | 1.001883 | ok | `2a4fdde` | DEPTH=10 (wider/deeper model, same aspect ratio) vs baseline DEPTH=8 |
| 10 | a5 | 1.045636 | ok | `2294c78` | DEPTH=6 (shallower/narrower model, more steps) vs baseline DEPTH=8 |
| 15 | a5 | 0.989057 | ok | `d164874` | TOTAL_BATCH_SIZE halved to 2^18 (more, noisier optimizer steps) vs baseline 2^19 |
| 18 | a5 | 0.999683 | ok | `060da74` | TOTAL_BATCH_SIZE further halved to 2^17 (DEVICE_BATCH_SIZE=64, grad_accum=1) vs winning 2^18 |
| 22 | a5 | 0.991229 | ok | `55814c1` | On top of winning 2^18 batch: MATRIX_LR 0.04->0.05 (more steps may tolerate higher Muon LR) |
| 26 | a5 | 0.991483 | ok | `877b32a` | On top of winning 2^18 batch (MATRIX_LR back to 0.04): WARMDOWN_RATIO 0.5->0.35 (more steps, shorter cooldown) |

**Best so far: val_bpb 0.989057 by a5 at commit `d164874`.**

## Findings (append-only broadcast)

- `#96` **a5** [commit `d164874`]: Halving TOTAL_BATCH_SIZE from 2^19 to 2^18 (DEPTH=8 unchanged, same DEVICE_BATCH_SIZE=128, grad_accum 1 instead of 2) improves val_bpb from baseline 0.997359 to 0.989057. More, noisier optimizer steps in the fixed 5-min budget outperforms fewer larger-batch steps here.
- `#129` **a5** [commit `d164874`]: Best result this share: val_bpb 0.989057 (vs baseline 0.997359), from halving TOTAL_BATCH_SIZE 2^19->2^18 with DEPTH=8 unchanged (grad_accum 2->1, same DEVICE_BATCH_SIZE=128). Tried further halving to 2^17 (worse, 0.999683), DEPTH=6/10 (both worse), MATRIX_LR 0.05 (worse, 0.991229), WARMDOWN_RATIO 0.35 (worse, 0.991483) -- none beat the plain batch-halving change. Simple, one-line change; recommend as new baseline for others.

## Disconfirmations (negative results, attempts to falsify)

- `#20` **a5** [commit `2a4fdde`]: DEPTH=10 (vs baseline DEPTH=8) gives val_bpb 1.001883, worse than baseline 0.997359. Deeper/wider model under fixed 5-min budget takes more time per step -> fewer optimizer steps, net negative here.
- `#53` **a5** [commit `2294c78`]: DEPTH=6 (vs baseline DEPTH=8) gives val_bpb 1.045636, much worse than baseline 0.997359. Combined with DEPTH=10 also being worse (1.001883, commit 2a4fdde), DEPTH=8 looks close to a local optimum for depth under this 5-min budget; not pursuing further depth changes.
- `#105` **a5** [commit `060da74`]: Further halving to TOTAL_BATCH_SIZE=2^17 (DEVICE_BATCH_SIZE=64) gives val_bpb 0.999683, worse than the 2^18 result (0.989057, commit d164874) and close to baseline. So 2^18 is a sweet spot, not 'smaller is always better' -- too-small batch reintroduces gradient noise that hurts.
- `#115` **a5** [commit `55814c1`]: On top of the winning 2^18 batch config, raising MATRIX_LR 0.04->0.05 gives val_bpb 0.991229, slightly worse than 0.04's 0.989057 (commit d164874). Muon LR was already well-tuned; batch size was the lever that mattered, not LR.
- `#128` **a5** [commit `877b32a`]: On top of the winning 2^18 batch config, shortening WARMDOWN_RATIO 0.5->0.35 gives val_bpb 0.991483, slightly worse than the unmodified schedule's 0.989057 (commit d164874). The 0.5 warmdown ratio was already well-tuned for the new step count.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

