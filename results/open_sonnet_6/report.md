# open_sonnet_6

## Setup

- protocol: open (Park et al. 2609.21032), shared log ON, share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-sonnet-5
- effort: medium
- run budget: 36 of 36 used (3.08 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: 42894f3a2466

## Result

- rounds completed: 1
- training runs: 36 (3.08 GPU-h used)
- val_bpb: 0.997333 -> 0.985651
- improvement: 1.17%
- agent cost: $1223.50
- wall clock: 6.02 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 5 | 0.985862 | 6 | 0 | 1 | 1 | $221.05 |
| a1 | 6 | 6 | 0.986182 | 2 | 2 | 1 | 1 | $236.01 |
| a2 | 6 | 4 | 0.985651 | 2 | 3 | 1 | 1 | $381.56 |
| a3 | 6 | 6 | 0.986059 | 2 | 3 | 1 | 2 | $199.46 |
| a4 | 6 | 6 | 0.987594 | 3 | 4 | 2 | 1 | $183.92 |
| a5 | 6 | 6 | 0.987994 | 3 | 3 | 1 | 1 | $1.51 |

Approaches claimed: a2: model shape scaling; a4: depth/width scaling sweep; a5: depth/width scaling + LR retune; a3: model depth/width scaling; a1: model scale & batch size tuning; a4: optimizer/schedule hyperparameters; a3: optimization schedule & LR/WD hyperparameters; a5: optimizer & schedule tuning; a0: optimizer & schedule tuning; a4: architecture micro-changes; a5: attention/MLP micro-architecture

Findings labelled weak: 1 of 18. Coordination notes: 5.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 16 | 2 | 0.987594 | a4/v0 |
| 0 | 32 | 3 | 0.985651 | a2/v0 |
| 0 | 35 | 3 | 0.985651 | a2/v0 |
| 0 | 36 | 3 | 0.985651 | a2/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 11 | 0.2072 |

## Errors

- crash rate: 8.3% (3/36)
- GPU spent on failed runs: 51s
- repair: {'slots_that_retried': 2, 'retries_that_recovered': 2, 'mean_attempts': 1.06}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 18, disconfirmations: 15, coordinations: 5, adoptions: 7, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 5 | 0 | $0.00 |
| a1 | 6 | 6 | 0 | $0.00 |
| a2 | 6 | 4 | 3 | $0.00 |
| a3 | 6 | 6 | 0 | $0.00 |
| a4 | 6 | 6 | 1 | $0.00 |
| a5 | 6 | 6 | 0 | $0.00 |
