# open_haiku_3

## Setup

- protocol: open (Park et al. 2609.21032), shared log ON, share per agent 12 runs
- agents: 3 x BoN=1 = 3 candidates/round
- models: claude-haiku-4-5-20251001
- effort: medium
- run budget: 36 of 36 used (3.36 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.280 (Claude Code)
- config fingerprint: 5b0d4f8b89e6

## Result

- rounds completed: 1
- training runs: 36 (3.36 GPU-h used)
- val_bpb: 0.996598 -> 0.990882
- improvement: 0.57%
- agent cost: $4.61
- wall clock: 3.38 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 12 | 12 | 0.990882 | 2 | 3 | 0 | 1 | $0.52 |
| a1 | 12 | 12 | 0.994809 | 1 | 0 | 0 | 1 | $0.90 |
| a2 | 12 | 12 | 0.994254 | 2 | 0 | 0 | 2 | $3.20 |

Approaches claimed: a1: model_architecture; a0: architecture; a2: optimizer

Findings labelled weak: 0 of 5. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 36 | 0 | 0.990882 | a0/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 3 | 0.1126 |

## Errors

- crash rate: 0.0% (0/36)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 5, disconfirmations: 3, coordinations: 0, adoptions: 0, messages: 0
agents that published at least one finding: 3 of 3

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 12 | 12 | 1 | $0.00 |
| a1 | 12 | 12 | 0 | $0.00 |
| a2 | 12 | 12 | 0 | $0.00 |
