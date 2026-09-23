# haiku_6

## Setup

- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-haiku-4-5-20251001
- effort: medium
- run budget: 28 of 36 used (2.15 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.280 (Claude Code)
- config fingerprint: b32649e17c7f

## Result

- rounds completed: 6
- training runs: 28 (2.15 GPU-h used)
- val_bpb: 0.996598 -> 0.994321
- improvement: 0.23%
- agent cost: $27.37
- wall clock: 2.78 h
- stopped because: round cap reached

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 6 | 1 | 0.996598 | — |
| 1 | 6 | 1 | 0.995700 | a2/v0 |
| 2 | 6 | 0 | 0.995700 | — |
| 3 | 6 | 0 | 0.995688 | a5/v0 |
| 4 | 6 | 1 | 0.994321 | a2/v0 |
| 5 | 6 | 0 | 0.994321 | — |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.1322 |
| 1 | 6 | 0.0900 |
| 2 | 6 | 0.3256 |
| 3 | 6 | 0.2170 |
| 4 | 6 | 0.2322 |
| 5 | 0 | n/a |

Mean similarity to the previous round's *other* agents: **0.2429**

## Errors

- crash rate: 8.3% (3/36)
- GPU spent on failed runs: 56s
- repair: {'slots_that_retried': 4, 'retries_that_recovered': 4, 'mean_attempts': 1.27}

No crashed idea was echoed by another agent in the next round.

## Use of the shared log

- unprompted messages written by agents: 17
- of those, threaded replies to a specific entry: 0
- rounds with at least one message: 5 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 6 | 0 | $1.70 |
| a1 | 6 | 5 | 0 | $1.67 |
| a2 | 6 | 6 | 2 | $1.88 |
| a3 | 6 | 5 | 0 | $1.28 |
| a4 | 6 | 5 | 0 | $2.05 |
| a5 | 6 | 6 | 1 | $1.82 |
