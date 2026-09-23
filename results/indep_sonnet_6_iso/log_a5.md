# Shared research directory (open protocol)

_You are `a5`. This cell runs agents independently: you see only your own entries and results. There are no peers to read or reach._

Baseline at start of the cell: **val_bpb 1.012347** (commit 228791f)

**Training runs remaining in this cell: 0 of 36.**
Each agent's share is 6 runs. Used so far: a0: 0 left, a1: 0 left, a2: 0 left, a3: 0 left, a4: 0 left, a5: 0 left.

## Approaches (slots)

- `#2` **a5** — depth/width scaling: Sweep DEPTH (and derived width via ASPECT_RATIO) to find the compute-optimal shape for a fixed 5-minute budget; will not touch optimizer internals (Muon/NorMuon) or value-embedding mechanics.

## Score log (one line per attempt, in GPU order)

| # | agent | val_bpb | status | commit | attempt |
|---|---|---|---|---|---|
| 1 | a5 | 1.001629 | ok | `2b65202` | DEPTH=10 (up from 8), width auto-scaled via ASPECT_RATIO |
| 6 | a5 | — | crash | `ed9df4b` | DEPTH=12, continuing depth sweep |
| 11 | a5 | 1.015093 | ok | `026e0b4` | DEPTH=12, DEVICE_BATCH_SIZE=64 to fix OOM |
| 16 | a5 | — | crash | `aec5df3` | DEPTH=11, DEVICE_BATCH_SIZE=128, narrowing depth sweep between 10 and 12 |
| 21 | a5 | 1.011977 | ok | `69b8e82` | DEPTH=11, DEVICE_BATCH_SIZE=64 to fix OOM, narrowing depth sweep |
| 26 | a5 | 1.002182 | ok | `9cb7d8b` | DEPTH=10 (best shape) + WARMDOWN_RATIO=0.3 (shorter cooldown, more time at full LR) |

**Best so far: val_bpb 1.001629 by a5 at commit `2b65202`.**

## Findings (append-only broadcast)

- `#8` **a5** [commit `2b65202`]: DEPTH=10 (n_embd scales via ASPECT_RATIO=64, head_dim=128) beats baseline DEPTH=8: val_bpb 1.001629 vs 1.012347 baseline. VRAM 68GB, well within budget.
- `#49` **a5** [commit `2b65202`]: Final result for a5: best is DEPTH=10 (n_embd=640 via ASPECT_RATIO=64/HEAD_DIM=128), DEVICE_BATCH_SIZE=128, all else default -> val_bpb 1.001629, beating baseline 1.012347. Depth sweep: 8(baseline)=1.012347, 10=1.001629 (best), 11(batch64, OOM-forced)=1.011977, 12(batch64)=1.015093. Going deeper than 10 requires shrinking DEVICE_BATCH_SIZE to avoid OOM, and that throughput/batch-size loss outweighs the extra depth's benefit within the 5-min budget. A small WARMDOWN_RATIO=0.3 tweak at DEPTH=10 gave 1.002182, no better than default 0.5 (noise-level).

## Disconfirmations (negative results, attempts to falsify)

- `#24` **a5** [commit `026e0b4`]: DEPTH=12 (with DEVICE_BATCH_SIZE reduced to 64 to avoid OOM) is worse than baseline: val_bpb 1.015093, worse than DEPTH=10's 1.001629 and roughly baseline DEPTH=8 (1.012347). Depth=10 looks like a local sweet spot for this 5-min budget; going deeper costs too much in batch size/throughput.

## Adoption events

_No adoptions yet._

## Coordination (conventions agreed after collisions)

_Nothing yet._

## Messages

_No messages._

