# haiku_3

## Setup

- agents: 3 x BoN=2 = 6 candidates/round
- models: claude-haiku-4-5-20251001
- effort: medium
- run budget: 21 of 36 used (1.78 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.280 (Claude Code)
- config fingerprint: 3cccd531fe97

## Result

- rounds completed: 6
- training runs: 21 (1.78 GPU-h used)
- val_bpb: 0.996598 -> 0.993742
- improvement: 0.29%
- agent cost: $21.49
- wall clock: 2.52 h
- stopped because: round cap reached

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 6 | 3 | 0.996598 | — |
| 1 | 6 | 3 | 0.996553 | a1/v0 |
| 2 | 6 | 0 | 0.996553 | — |
| 3 | 6 | 1 | 0.996221 | a1/v0 |
| 4 | 6 | 0 | 0.995001 | a0/v1 |
| 5 | 6 | 0 | 0.993742 | a0/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 3 | 0.1448 |
| 1 | 3 | 0.1462 |
| 2 | 3 | 0.2008 |
| 3 | 3 | 0.4550 |
| 4 | 3 | 0.1755 |
| 5 | 3 | 0.1743 |

Mean similarity to the previous round's *other* agents: **0.1921**

## Errors

- crash rate: 19.4% (7/36)
- GPU spent on failed runs: 32s
- repair: {'slots_that_retried': 2, 'retries_that_recovered': 2, 'mean_attempts': 1.11}

No crashed idea was echoed by another agent in the next round.

## Use of the shared log

- unprompted messages written by agents: 25
- of those, threaded replies to a specific entry: 0
- rounds with at least one message: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 12 | 8 | 2 | $3.29 |
| a1 | 12 | 11 | 2 | $4.46 |
| a2 | 12 | 10 | 0 | $3.50 |
