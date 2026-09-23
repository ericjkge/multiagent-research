# open_opus_1

## Setup

- protocol: open (Park et al. 2609.21032), shared log ON, share per agent 36 runs
- agents: 1 x BoN=1 = 1 candidates/round
- models: claude-opus-5
- effort: medium
- run budget: 36 of 36 used (3.42 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: a6ee3ff26ccd

## Result

- rounds completed: 1
- training runs: 36 (3.42 GPU-h used)
- val_bpb: 0.997359 -> 0.980742
- improvement: 1.67%
- agent cost: $5.48
- wall clock: 3.63 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 36 | 36 | 0.980742 | 10 | 16 | 0 | 1 | $5.48 |

Approaches claimed: a0: compute-allocation co-design

Findings labelled weak: 0 of 10. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 36 | 0 | 0.980742 | a0/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 1 | n/a |

## Errors

- crash rate: 0.0% (0/36)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 10, disconfirmations: 16, coordinations: 0, adoptions: 0, messages: 0
agents that published at least one finding: 1 of 1

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 36 | 36 | 1 | $0.00 |
