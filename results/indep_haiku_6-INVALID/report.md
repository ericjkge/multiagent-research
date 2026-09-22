# indep_haiku_6

## Setup

- protocol: open (Park et al. 2609.21032), shared log OFF (independent agents), share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-haiku-4-5-20251001
- effort: medium
- run budget: 13 of 36 used (1.11 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: 5334d3a5cc9d

## Result

- rounds completed: 1
- training runs: 13 (1.11 GPU-h used)
- val_bpb: 0.996937 -> 0.996838
- improvement: 0.01%
- agent cost: $11.91
- wall clock: 1.21 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 1 | 1 | 0.997722 | 2 | 1 | 0 | 1 | $1.30 |
| a1 | 2 | 2 | 0.996838 | 1 | 0 | 0 | 1 | $1.08 |
| a2 | 1 | 1 | 1.002563 | 1 | 1 | 0 | 1 | $1.50 |
| a3 | 0 | 0 | — | 3 | 0 | 0 | 3 | $6.16 |
| a4 | 2 | 2 | 0.999534 | 1 | 2 | 0 | 1 | $1.11 |
| a5 | 6 | 6 | 0.996937 | 7 | 1 | 0 | 1 | $0.76 |

Approaches claimed: a4: hyperparameter-tuning; a5: optimizer_lr_tuning; a1: hyperparameter-tuning; a0: learning-rate-schedules; a2: depth-width-lr-tradeoffs; a3: depth_scaling

Findings labelled weak: 8 of 15. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 12 | 0 | 0.996838 | a1/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.1704 |

## Errors

- crash rate: 0.0% (0/12)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 15, disconfirmations: 5, coordinations: 0, adoptions: 0, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 1 | 1 | 0 | $0.00 |
| a1 | 2 | 2 | 1 | $0.00 |
| a2 | 1 | 1 | 0 | $0.00 |
| a4 | 2 | 2 | 0 | $0.00 |
| a5 | 6 | 6 | 0 | $0.00 |
