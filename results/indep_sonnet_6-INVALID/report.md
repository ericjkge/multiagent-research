# indep_sonnet_6

## Setup

- protocol: open (Park et al. 2609.21032), shared log OFF (independent agents), share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-sonnet-5
- effort: medium
- run budget: 32 of 36 used (2.64 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: 7a1df42e0255

## Result

- rounds completed: 1
- training runs: 32 (2.64 GPU-h used)
- val_bpb: 0.997333 -> 0.988312
- improvement: 0.90%
- agent cost: $74.56
- wall clock: 2.72 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 6 | 0.995760 | 3 | 2 | 0 | 2 | $2.56 |
| a1 | 6 | 6 | 0.994598 | 2 | 3 | 0 | 11 | $20.74 |
| a2 | 6 | 6 | 0.988312 | 2 | 3 | 0 | 1 | $1.63 |
| a3 | 1 | 1 | 1.042821 | 1 | 0 | 0 | 13 | $24.24 |
| a4 | 6 | 5 | 1.003372 | 1 | 5 | 0 | 9 | $18.69 |
| a5 | 5 | 4 | 0.997949 | 3 | 1 | 0 | 5 | $6.70 |

Approaches claimed: a4: depth/width scaling; a1: model sizing / depth-width tradeoff; a3: depth/width scaling + schedule tuning; a0: compute-optimal depth/width scan; a2: depth/width scaling + batch size tradeoff; a5: model-size-scaling

Findings labelled weak: 2 of 12. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 30 | 2 | 0.988312 | a2/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.2934 |

## Errors

- crash rate: 6.7% (2/30)
- GPU spent on failed runs: 34s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 12, disconfirmations: 14, coordinations: 0, adoptions: 0, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 6 | 0 | $0.00 |
| a1 | 6 | 6 | 0 | $0.00 |
| a2 | 6 | 6 | 1 | $0.00 |
| a3 | 1 | 1 | 0 | $0.00 |
| a4 | 6 | 5 | 0 | $0.00 |
| a5 | 5 | 4 | 0 | $0.00 |
