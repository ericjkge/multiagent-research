# open_sonnet_3

## Setup

- protocol: open (Park et al. 2609.21032), shared log ON, share per agent 12 runs
- agents: 3 x BoN=1 = 3 candidates/round
- models: claude-sonnet-5
- effort: medium
- run budget: 36 of 36 used (3.43 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.280 (Claude Code)
- config fingerprint: 07d05f023fde

## Result

- rounds completed: 1
- training runs: 36 (3.43 GPU-h used)
- val_bpb: 0.995733 -> 0.993862
- improvement: 0.19%
- agent cost: $48.71
- wall clock: 3.45 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 12 | 12 | 0.994013 | 3 | 5 | 2 | 8 | $21.39 |
| a1 | 12 | 12 | 0.994317 | 5 | 2 | 2 | 4 | $9.58 |
| a2 | 12 | 12 | 0.993862 | 5 | 6 | 0 | 5 | $17.74 |

Approaches claimed: a1: model scale & batch-size tuning; a0: depth/width/batch scaling; a2: compute-allocation-scaling; a1: optimizer & LR-schedule tuning; a2: optimizer-schedule-tuning; a0: optimizer & schedule tuning

Findings labelled weak: 1 of 13. Coordination notes: 5.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 36 | 0 | 0.993862 | a2/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.2529 |

## Errors

- crash rate: 0.0% (0/36)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 13, disconfirmations: 13, coordinations: 5, adoptions: 4, messages: 0
agents that published at least one finding: 3 of 3

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 12 | 12 | 0 | $0.00 |
| a1 | 12 | 12 | 0 | $0.00 |
| a2 | 12 | 12 | 1 | $0.00 |
