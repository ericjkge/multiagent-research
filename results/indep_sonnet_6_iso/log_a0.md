# Shared research directory (open protocol)

_You are `a0`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#5` **a0** — depth/width scaling: Will search DEPTH and ASPECT_RATIO (model width) to find the best compute-optimal shape for the fixed 5-minute budget, holding the existing optimizer/architecture (Muon+AdamW, value embeddings, sliding window) fixed. Will not assume the default DEPTH=8 is optimal for this time budget; will not touch prepare.py or eval.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 5 | a0 | 1.063523 | ok | `00e4480` | DEPTH=6 (narrower/shallower, more steps in 5min) |
| 10 | a0 | 1.001260 | ok | `6b380f8` | DEPTH=10 (deeper/wider, fewer steps in 5min) |
| 15 | a0 | — | crash | `c32f9fd` | DEPTH=12 (even deeper/wider) |
| 20 | a0 | 1.015048 | ok | `6444955` | DEPTH=12 with DEVICE_BATCH_SIZE=64 to fix OOM |
| 25 | a0 | 0.998370 | ok | `c41c30a` | DEPTH=9 (same width 640 as DEPTH=10, one fewer layer, more steps) |
| 30 | a0 | 0.996232 | ok | `d77862a` | DEPTH=8, ASPECT_RATIO=80 -> width=640 (isolate depth vs width vs D9/D10) |

**Best so far: val_bpb 0.996232 by a0 at commit `d77862a`.**

## Findings (append-only broadcast)

- `#15` **a0** [commit `00e4480`]: DEPTH=6 (dim=384) worse than baseline DEPTH=8: val_bpb 1.063523 vs baseline 1.012347. Shallower/narrower net hurts more than the extra steps help in 5min budget.
- `#22` **a0** [commit `6b380f8`]: DEPTH=10 (dim=640) beats baseline DEPTH=8: val_bpb 1.001260 vs 1.012347 baseline. Deeper/wider model wins despite fewer steps in the 5min budget.
- `#47` **a0** [commit `c41c30a`]: New best: DEPTH=9 (dim=640, same width as DEPTH=10) val_bpb 0.998370, beats DEPTH=10 (1.001260) and baseline DEPTH=8 (1.012347). Depth sweep at fixed 5min budget: D6=1.0635, D8(baseline,dim512)=1.0123, D9(dim640)=0.9984, D10(dim640)=1.0013, D12(dim768)=1.0150 -- best is D9 with width 640, i.e. slightly fewer layers than D10 at the same width trains faster per step and wins in the fixed time budget.
- `#64` **a0** [commit `d77862a`]: Final best: DEPTH=8 with ASPECT_RATIO=80 (width=640, same width as D9/D10 runs) val_bpb 0.996232 -- beats D9@640 (0.998370) and D10@640 (1.001260). So at fixed width 640, DEPTH=8 > 9 > 10: shallower is better once width is controlled for. Conclusion: the D9 gain over baseline D8(width512) was mostly a WIDTH effect, not depth; widening from 512->640 while keeping depth=8 gives the best result of my sweep. Full depth/width sweep (5min budget): D6/w384=1.0635, D8/w512(baseline)=1.0123, D8/w640=0.9962(best), D9/w640=0.9984, D10/w640=1.0013, D12/w768=1.0150.

## Disconfirmations (negative results, attempts to falsify)

- `#40` **a0** [commit `6444955`]: DEPTH=12 (dim=768) worse than DEPTH=10: val_bpb 1.015048 vs 1.001260, roughly back to baseline. Depth sweet spot for this 5min budget is near DEPTH=10, not monotonically improving with depth. Note: DEPTH=12 at DEVICE_BATCH_SIZE=128 OOMs (78GB+), needed DEVICE_BATCH_SIZE=64.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

