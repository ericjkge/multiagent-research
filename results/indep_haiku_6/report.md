# indep_haiku_6

## Setup

- protocol: open (Park et al. 2609.21032), shared log OFF (independent agents), share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-haiku-4-5-20251001
- effort: medium
- run budget: 36 of 36 used (2.44 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: 5334d3a5cc9d

## Result

- rounds completed: 1
- training runs: 36 (2.44 GPU-h used)
- val_bpb: 0.993369 -> 0.976806
- improvement: 1.67%
- agent cost: $23.33
- wall clock: 5.49 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 4 | 0.976806 | 4 | 1 | 0 | 1 | $3.24 |
| a1 | 6 | 3 | 0.980185 | 3 | 0 | 0 | 1 | $2.61 |
| a2 | 6 | 3 | 0.982450 | 4 | 1 | 0 | 1 | $3.74 |
| a3 | 6 | 5 | 0.978964 | 7 | 1 | 0 | 1 | $9.30 |
| a4 | 6 | 4 | 0.979237 | 3 | 2 | 0 | 1 | $2.75 |
| a5 | 6 | 6 | 0.996937 | 7 | 1 | 0 | 1 | $1.69 |

Approaches claimed: a4: hyperparameter-tuning; a5: optimizer_lr_tuning; a1: hyperparameter-tuning; a0: learning-rate-schedules; a2: depth-width-lr-tradeoffs; a3: depth_scaling; a3: empirical_stack

Findings labelled weak: 8 of 28. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 12 | 0 | 0.996838 | a1/v0 |
| 0 | 36 | 11 | 0.976806 | a0/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 7 | 0.1458 |

## Errors

- crash rate: 30.6% (11/36)
- GPU spent on failed runs: 93s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 28, disconfirmations: 6, coordinations: 0, adoptions: 0, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 4 | 1 | $0.00 |
| a1 | 6 | 3 | 1 | $0.00 |
| a2 | 6 | 3 | 0 | $0.00 |
| a3 | 6 | 5 | 0 | $0.00 |
| a4 | 6 | 4 | 0 | $0.00 |
| a5 | 6 | 6 | 0 | $0.00 |
