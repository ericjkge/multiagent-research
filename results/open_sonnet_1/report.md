# open_sonnet_1

## Setup

- protocol: open (Park et al. 2609.21032), shared log ON, share per agent 36 runs
- agents: 1 x BoN=1 = 1 candidates/round
- models: claude-sonnet-5
- effort: medium
- run budget: 36 of 36 used (3.44 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.280 (Claude Code)
- config fingerprint: d2f94edd872a

## Result

- rounds completed: 1
- training runs: 36 (3.44 GPU-h used)
- val_bpb: 1.012531 -> 0.979324
- improvement: 3.28%
- agent cost: $12.30
- wall clock: 3.57 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 36 | 35 | 0.979324 | 9 | 0 | 0 | 2 | $12.30 |

Approaches claimed: a0: depth/width & LR scaling

Findings labelled weak: 0 of 9. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 36 | 1 | 0.979324 | a0/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 1 | n/a |

## Errors

- crash rate: 2.8% (1/36)
- GPU spent on failed runs: 24s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 9, disconfirmations: 0, coordinations: 0, adoptions: 0, messages: 0
agents that published at least one finding: 1 of 1

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 36 | 35 | 1 | $0.00 |
