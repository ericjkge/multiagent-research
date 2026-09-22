# opus_6

## Setup

- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-opus-5
- effort: medium
- run budget: 29 of 36 used (2.77 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: cdeacad80dd2

## Result

- rounds completed: 5
- training runs: 29 (2.77 GPU-h used)
- val_bpb: 0.997280 -> 0.981817
- improvement: 1.55%
- agent cost: $291.43
- wall clock: 3.44 h
- stopped because: agent cost ceiling reached

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 6 | 0 | 0.988076 | a5/v0 |
| 1 | 6 | 0 | 0.988076 | — |
| 2 | 6 | 0 | 0.988076 | — |
| 3 | 6 | 0 | 0.987227 | a0/v0 |
| 4 | 6 | 0 | 0.981817 | a1/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.2465 |
| 1 | 6 | 0.3382 |
| 2 | 6 | 0.4105 |
| 3 | 6 | 0.3499 |
| 4 | 6 | 0.2435 |

Mean similarity to the previous round's *other* agents: **0.2293**

## Errors

- crash rate: 0.0% (0/30)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared log

- unprompted messages written by agents: 80
- of those, threaded replies to a specific entry: 44
- rounds with at least one message: 5 of 5

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 5 | 5 | 1 | $13.18 |
| a1 | 5 | 5 | 1 | $19.25 |
| a2 | 5 | 5 | 0 | $14.78 |
| a3 | 5 | 5 | 0 | $15.76 |
| a4 | 5 | 5 | 0 | $15.11 |
| a5 | 5 | 5 | 1 | $14.61 |
