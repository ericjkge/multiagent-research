# sonnet_3

## Setup

- agents: 3 x BoN=2 = 6 candidates/round
- models: claude-sonnet-5
- effort: medium
- run budget: 32 of 36 used (2.73 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: cbe33f50e854

## Result

- rounds completed: 5
- training runs: 32 (2.73 GPU-h used)
- val_bpb: 1.002652 -> 0.992950
- improvement: 0.97%
- agent cost: $28.86
- wall clock: 2.82 h
- stopped because: training-run budget exhausted (4 slots left, a round needs 6)

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 6 | 0 | 1.002530 | a0/v1 |
| 1 | 6 | 2 | 1.002530 | — |
| 2 | 6 | 0 | 0.998924 | a1/v1 |
| 3 | 6 | 0 | 0.998440 | a2/v1 |
| 4 | 6 | 0 | 0.992950 | a1/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 3 | 0.2776 |
| 1 | 3 | 0.2387 |
| 2 | 3 | 0.2521 |
| 3 | 3 | 0.5462 |
| 4 | 3 | 0.3095 |

Mean similarity to the previous round's *other* agents: **0.2572**

## Errors

- crash rate: 6.7% (2/30)
- GPU spent on failed runs: 71s
- repair: {'slots_that_retried': 2, 'retries_that_recovered': 2, 'mean_attempts': 1.07}

No crashed idea was echoed by another agent in the next round.

## Use of the shared log

Agents wrote **nothing** to the log beyond their required response. They had `arena-log` available and did not reach for it.

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 10 | 10 | 1 | $5.00 |
| a1 | 10 | 8 | 2 | $5.00 |
| a2 | 10 | 10 | 1 | $4.97 |
