# open_opus_3

## Setup

- protocol: open (Park et al. 2609.21032), shared log ON, share per agent 12 runs
- agents: 3 x BoN=1 = 3 candidates/round
- models: claude-opus-5
- effort: medium
- run budget: 36 of 36 used (3.32 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: 08a3aeb2da70

## Result

- rounds completed: 1
- training runs: 36 (3.32 GPU-h used)
- val_bpb: 0.997359 -> 0.982130
- improvement: 1.53%
- agent cost: $18.99
- wall clock: 3.34 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 12 | 11 | 0.983408 | 10 | 3 | 1 | 1 | $6.55 |
| a1 | 12 | 12 | 0.982130 | 6 | 6 | 2 | 1 | $5.98 |
| a2 | 12 | 12 | 0.982382 | 9 | 2 | 2 | 1 | $6.46 |

Approaches claimed: a0: model shape / compute allocation; a1: architecture efficiency (attention/MLP shape); a2: attention & positional structure

Findings labelled weak: 8 of 25. Coordination notes: 3.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 36 | 1 | 0.982130 | a1/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 3 | 0.1993 |

## Errors

- crash rate: 2.8% (1/36)
- GPU spent on failed runs: 19s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 25, disconfirmations: 11, coordinations: 3, adoptions: 5, messages: 0
agents that published at least one finding: 3 of 3

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 12 | 11 | 0 | $0.00 |
| a1 | 12 | 12 | 1 | $0.00 |
| a2 | 12 | 12 | 0 | $0.00 |
