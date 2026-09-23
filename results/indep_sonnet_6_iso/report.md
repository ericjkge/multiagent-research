# indep_sonnet_6_iso

## Setup

- protocol: open (Park et al. 2609.21032), shared log OFF (independent agents), share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-sonnet-5
- effort: medium
- run budget: 36 of 36 used (2.80 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.280 (Claude Code)
- config fingerprint: 1845a1eb0ee2

## Result

- rounds completed: 1
- training runs: 36 (2.80 GPU-h used)
- val_bpb: 1.001629 -> 0.996232
- improvement: 0.54%
- agent cost: $4.38
- wall clock: 2.85 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 5 | 0.996232 | 4 | 1 | 0 | 1 | $0.67 |
| a1 | 6 | 5 | 1.001799 | 2 | 1 | 0 | 1 | $0.78 |
| a2 | 6 | 5 | 0.997177 | 3 | 2 | 0 | 1 | $0.67 |
| a3 | 6 | 5 | 1.011362 | 4 | 2 | 0 | 1 | $0.64 |
| a4 | 6 | 5 | 1.012919 | 5 | 2 | 0 | 1 | $0.80 |
| a5 | 6 | 4 | 1.001629 | 2 | 1 | 0 | 1 | $0.82 |

Approaches claimed: a5: depth/width scaling; a1: depth/width & LR scan; a3: depth/width scaling for fixed time budget; a0: depth/width scaling; a4: compute-optimal depth/width under fixed 5-min wall-clock; a2: depth/width shape + LR schedule search

Findings labelled weak: 3 of 20. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 30 | 6 | 0.996232 | a0/v0 |
| 0 | 30 | 6 | 0.996232 | a0/v0 |
| 0 | 36 | 7 | 0.996232 | a0/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.3066 |

## Errors

- crash rate: 19.4% (7/36)
- GPU spent on failed runs: 103s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 20, disconfirmations: 9, coordinations: 0, adoptions: 0, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 5 | 3 | $0.00 |
| a1 | 6 | 5 | 0 | $0.00 |
| a2 | 6 | 5 | 0 | $0.00 |
| a3 | 6 | 5 | 0 | $0.00 |
| a4 | 6 | 5 | 0 | $0.00 |
| a5 | 6 | 4 | 0 | $0.00 |
