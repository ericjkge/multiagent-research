# indep_sonnet_6_s2

## Setup

- protocol: open (Park et al. 2609.21032), shared log OFF (independent agents), share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-sonnet-5
- effort: medium
- run budget: 36 of 36 used (2.81 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: f965d7990710

## Result

- rounds completed: 1
- training runs: 36 (2.81 GPU-h used)
- val_bpb: 0.997359 -> 0.977046
- improvement: 2.04%
- agent cost: $2077.04
- wall clock: 3.70 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 4 | 0.977046 | 4 | 3 | 0 | 5 | $1979.38 |
| a1 | 6 | 6 | 0.977190 | 4 | 2 | 0 | 1 | $9.79 |
| a2 | 6 | 4 | 0.990382 | 2 | 1 | 0 | 27 | $63.52 |
| a3 | 6 | 3 | 0.977903 | 2 | 2 | 0 | 12 | $21.49 |
| a4 | 6 | 5 | 0.997158 | 4 | 1 | 0 | 1 | $1.25 |
| a5 | 6 | 6 | 0.989057 | 2 | 5 | 0 | 1 | $1.60 |

Approaches claimed: a0: depth/width scaling + LR tuning; a3: compute-efficiency scaling; a1: depth/width scaling + LR schedule tuning; a4: model depth/width scaling under fixed 5-min budget; a2: hparam/schedule tuning; a5: depth/width and batch-size scaling

Findings labelled weak: 5 of 18. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 29 | 6 | 0.977903 | a3/v0 |
| 0 | 32 | 6 | 0.977046 | a0/v0 |
| 0 | 34 | 7 | 0.977046 | a0/v0 |
| 0 | 37 | 8 | 0.977046 | a0/v0 |
| 0 | 36 | 8 | 0.977046 | a0/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.2848 |

## Errors

- crash rate: 22.2% (8/36)
- GPU spent on failed runs: 455s
- repair: {'slots_that_retried': 1, 'retries_that_recovered': 1, 'mean_attempts': 1.03}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 18, disconfirmations: 14, coordinations: 0, adoptions: 0, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 4 | 4 | $0.00 |
| a1 | 6 | 6 | 0 | $0.00 |
| a2 | 6 | 4 | 0 | $0.00 |
| a3 | 6 | 3 | 1 | $0.00 |
| a4 | 6 | 5 | 0 | $0.00 |
| a5 | 6 | 6 | 0 | $0.00 |
