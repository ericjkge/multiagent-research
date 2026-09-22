# opus_3

## Setup

- agents: 3 x BoN=2 = 6 candidates/round
- models: claude-opus-5
- effort: medium
- run budget: 36 of 36 used (3.30 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: abd7f53ba799

## Result

- rounds completed: 6
- training runs: 36 (3.30 GPU-h used)
- val_bpb: 0.996770 -> 0.987792
- improvement: 0.90%
- agent cost: $235.32
- wall clock: 4.14 h
- stopped because: round cap reached

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 6 | 0 | 0.992664 | a2/v0 |
| 1 | 6 | 0 | 0.992664 | — |
| 2 | 6 | 0 | 0.989668 | a2/v0 |
| 3 | 6 | 1 | 0.988725 | a2/v1 |
| 4 | 6 | 0 | 0.987792 | a1/v1 |
| 5 | 6 | 1 | 0.987792 | — |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 3 | 0.3384 |
| 1 | 3 | 0.2384 |
| 2 | 3 | 0.2277 |
| 3 | 3 | 0.3684 |
| 4 | 3 | 0.1604 |
| 5 | 3 | 0.4029 |

Mean similarity to the previous round's *other* agents: **0.2444**

## Errors

- crash rate: 5.6% (2/36)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 1, 'retries_that_recovered': 1, 'mean_attempts': 1.03}

No crashed idea was echoed by another agent in the next round.

## Use of the shared log

- unprompted messages written by agents: 78
- of those, threaded replies to a specific entry: 10
- rounds with at least one message: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 12 | 12 | 0 | $40.53 |
| a1 | 12 | 10 | 1 | $49.87 |
| a2 | 12 | 12 | 3 | $34.60 |
