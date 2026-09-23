# sonnet_1

## Setup

- agents: 1 x BoN=5 = 5 candidates/round
- models: claude-sonnet-5
- effort: medium
- run budget: 32 of 36 used (2.89 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: 264edee22726

## Result

- rounds completed: 6
- training runs: 32 (2.89 GPU-h used)
- val_bpb: 1.002279 -> 1.002279
- improvement: 0.00%
- agent cost: $16.07
- wall clock: 2.97 h
- stopped because: training-run budget exhausted (4 slots left, a round needs 5)

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 5 | 0 | 1.002279 | a0/v0 |
| 1 | 5 | 1 | 1.002279 | — |
| 2 | 5 | 0 | 1.002279 | — |
| 3 | 5 | 0 | 1.002279 | — |
| 4 | 5 | 0 | 1.002279 | — |
| 5 | 5 | 0 | 1.002279 | — |

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

- crash rate: 3.3% (1/30)
- GPU spent on failed runs: 31s
- repair: {'slots_that_retried': 2, 'retries_that_recovered': 1, 'mean_attempts': 1.07}

No crashed idea was echoed by another agent in the next round.

## Use of the shared log

Agents wrote **nothing** to the log beyond their required response. They had `arena-log` available and did not reach for it.

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 30 | 29 | 1 | $15.03 |
