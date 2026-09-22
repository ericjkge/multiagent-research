# opus_6

## Setup

- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-opus-5
- effort: medium
- run budget: 34 of 36 used (3.30 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: cdeacad80dd2

## Result

- rounds completed: 6
- training runs: 34 (3.30 GPU-h used)
- val_bpb: 0.997280 -> 0.980496
- improvement: 1.68%
- agent cost: $407.50
- wall clock: 10.37 h
- stopped because: round cap reached

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 6 | 0 | 0.988076 | a5/v0 |
| 1 | 6 | 0 | 0.988076 | — |
| 2 | 6 | 0 | 0.988076 | — |
| 3 | 6 | 0 | 0.987227 | a0/v0 |
| 4 | 6 | 0 | 0.981817 | a1/v0 |
| 5 | 6 | 0 | 0.980496 | a1/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.2465 |
| 1 | 6 | 0.3382 |
| 2 | 6 | 0.4105 |
| 3 | 6 | 0.3499 |
| 4 | 6 | 0.2435 |
| 5 | 6 | 0.1938 |

Mean similarity to the previous round's *other* agents: **0.2209**

## Errors

- crash rate: 0.0% (0/36)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared log

- unprompted messages written by agents: 95
- of those, threaded replies to a specific entry: 53
- rounds with at least one message: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 6 | 1 | $20.00 |
| a1 | 6 | 6 | 2 | $24.64 |
| a2 | 6 | 6 | 0 | $20.35 |
| a3 | 6 | 6 | 0 | $21.72 |
| a4 | 6 | 6 | 0 | $22.59 |
| a5 | 6 | 6 | 1 | $20.17 |
