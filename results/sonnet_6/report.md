# sonnet_6

## Setup

- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-sonnet-5
- effort: medium
- run budget: 33 of 36 used (2.80 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: 0cf7ca994861

## Result

- rounds completed: 5
- training runs: 33 (2.80 GPU-h used)
- val_bpb: 1.001440 -> 0.978041
- improvement: 2.34%
- agent cost: $61.22
- wall clock: 2.92 h
- stopped because: training-run budget exhausted (3 slots left, a round needs 6)

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 6 | 0 | 0.998671 | a2/v0 |
| 1 | 6 | 0 | 0.998671 | — |
| 2 | 6 | 0 | 0.982558 | a5/v0 |
| 3 | 6 | 0 | 0.982306 | a2/v0 |
| 4 | 6 | 0 | 0.978041 | a4/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.2496 |
| 1 | 6 | 0.4710 |
| 2 | 6 | 0.4964 |
| 3 | 6 | 0.1862 |
| 4 | 6 | 0.1964 |

Mean similarity to the previous round's *other* agents: **0.2824**

## Errors

- crash rate: 0.0% (0/30)
- GPU spent on failed runs: 59s
- repair: {'slots_that_retried': 4, 'retries_that_recovered': 4, 'mean_attempts': 1.14}

No crashed idea was echoed by another agent in the next round.

## Use of the shared log

- unprompted messages written by agents: 2
- of those, threaded replies to a specific entry: 0
- rounds with at least one message: 2 of 5

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 5 | 5 | 0 | $3.64 |
| a1 | 5 | 5 | 0 | $3.50 |
| a2 | 5 | 5 | 2 | $3.55 |
| a3 | 5 | 5 | 0 | $3.23 |
| a4 | 5 | 5 | 1 | $3.33 |
| a5 | 5 | 5 | 1 | $3.35 |
