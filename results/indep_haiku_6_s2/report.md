# indep_haiku_6_s2

## Setup

- protocol: open (Park et al. 2609.21032), shared log OFF (independent agents), share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-haiku-4-5-20251001
- effort: medium
- run budget: 36 of 36 used (3.31 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: e86b05f80cef

## Result

- rounds completed: 1
- training runs: 36 (3.31 GPU-h used)
- val_bpb: 0.997333 -> 0.995414
- improvement: 0.19%
- agent cost: $16.33
- wall clock: 3.34 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 6 | 0.996295 | 5 | 3 | 0 | 3 | $3.94 |
| a1 | 6 | 6 | 0.996103 | 2 | 4 | 0 | 2 | $1.86 |
| a2 | 6 | 6 | 0.996052 | 4 | 6 | 0 | 2 | $2.57 |
| a3 | 6 | 6 | 0.996858 | 6 | 0 | 0 | 1 | $4.13 |
| a4 | 6 | 6 | 0.995414 | 1 | 0 | 0 | 1 | $1.46 |
| a5 | 6 | 6 | 0.996370 | 4 | 1 | 0 | 3 | $2.37 |

Approaches claimed: a0: capacity_and_optimization; a1: optimizer_and_lr_tuning; a2: learning_rate_exploration; a3: learning_rate_tuning; a5: optimizer_tuning; a4: learning_rate_schedule

Findings labelled weak: 1 of 22. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 35 | 0 | 0.995414 | a4/v0 |
| 0 | 36 | 0 | 0.995414 | a4/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.1523 |

## Errors

- crash rate: 0.0% (0/36)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 22, disconfirmations: 14, coordinations: 0, adoptions: 0, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 6 | 0 | $0.00 |
| a1 | 6 | 6 | 0 | $0.00 |
| a2 | 6 | 6 | 0 | $0.00 |
| a3 | 6 | 6 | 0 | $0.00 |
| a4 | 6 | 6 | 2 | $0.00 |
| a5 | 6 | 6 | 0 | $0.00 |
