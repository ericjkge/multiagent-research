# opus_1

## Setup

- agents: 1 x BoN=3 = 3 candidates/round
- models: claude-opus-5
- effort: medium
- run budget: 36 of 36 used (3.36 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: 3c97f93b1b50

## Result

- rounds completed: 8
- training runs: 36 (3.36 GPU-h used)
- val_bpb: 0.997359 -> 0.986277
- improvement: 1.11%
- agent cost: $69.29
- wall clock: 11.38 h
- stopped because: training-run budget exhausted (0 slots left, a round needs 3)

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 5 | 1 | 0.997358 | a0/v2 |
| 1 | 5 | 0 | 0.997358 | — |
| 2 | 5 | 0 | 0.989007 | a0/v0 |
| 3 | 5 | 0 | 0.986277 | a0/v3 |
| 4 | 5 | 0 | 0.986277 | — |
| 5 | 5 | 0 | 0.986277 | — |
| 6 | 5 | 0 | 0.986277 | — |
| 7 | 3 | 0 | 0.986277 | — |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 1 | n/a |
| 1 | 1 | n/a |
| 2 | 1 | n/a |
| 3 | 1 | n/a |
| 4 | 1 | n/a |
| 5 | 1 | n/a |
| 6 | 1 | n/a |
| 7 | 1 | n/a |

## Errors

- crash rate: 2.6% (1/38)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared log

- unprompted messages written by agents: 35
- of those, threaded replies to a specific entry: 8
- rounds with at least one message: 8 of 8

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 38 | 37 | 3 | $65.79 |
