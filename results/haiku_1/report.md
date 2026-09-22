# haiku_1

## Setup

- agents: 1 x BoN=6 = 6 candidates/round
- models: claude-haiku-4-5-20251001
- effort: medium
- run budget: 20 of 36 used (1.85 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.280 (Claude Code)
- config fingerprint: a07c0fa1cd36

## Result

- rounds completed: 6
- training runs: 20 (1.85 GPU-h used)
- val_bpb: 0.996598 -> 0.994120
- improvement: 0.25%
- agent cost: $10.09
- wall clock: 2.46 h
- stopped because: round cap reached

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 6 | 2 | 0.996598 | — |
| 1 | 6 | 3 | 0.996598 | — |
| 2 | 6 | 0 | 0.996000 | a0/v2 |
| 3 | 6 | 2 | 0.996000 | — |
| 4 | 6 | 1 | 0.996000 | — |
| 5 | 6 | 0 | 0.994120 | a0/v1 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 1 | n/a |
| 1 | 1 | n/a |
| 2 | 1 | n/a |
| 3 | 1 | n/a |
| 4 | 1 | n/a |
| 5 | 1 | n/a |

## Errors

- crash rate: 22.2% (8/36)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared log

- unprompted messages written by agents: 23
- of those, threaded replies to a specific entry: 0
- rounds with at least one message: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 36 | 28 | 2 | $9.72 |
