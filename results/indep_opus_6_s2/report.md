# indep_opus_6_s2

## Setup

- protocol: open (Park et al. 2609.21032), shared log OFF (independent agents), share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-opus-5
- effort: medium
- run budget: 36 of 36 used (3.38 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: 700b542e3d57

## Result

- rounds completed: 1
- training runs: 36 (3.38 GPU-h used)
- val_bpb: 0.987686 -> 0.980071
- improvement: 0.77%
- agent cost: $32.99
- wall clock: 3.89 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 6 | 0.981817 | 4 | 3 | 0 | 1 | $3.73 |
| a1 | 6 | 6 | 0.985236 | 4 | 4 | 0 | 1 | $5.34 |
| a2 | 6 | 6 | 0.984640 | 4 | 5 | 0 | 1 | $5.51 |
| a3 | 6 | 6 | 0.982139 | 5 | 2 | 0 | 1 | $6.63 |
| a4 | 6 | 6 | 0.980071 | 3 | 3 | 0 | 1 | $7.67 |
| a5 | 6 | 6 | 0.984380 | 3 | 4 | 0 | 1 | $4.11 |

Approaches claimed: a1: batch-size & step-count economics; a0: compute allocation: batch size x model shape; a5: model shape & attention head geometry; a4: attention microarchitecture; a2: batch-size / step-count allocation; a3: compute allocation: batch size x model shape

Findings labelled weak: 1 of 23. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 36 | 0 | 0.980071 | a4/v0 |
| 0 | 36 | 0 | 0.980071 | a4/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.2344 |

## Errors

- crash rate: 0.0% (0/36)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 23, disconfirmations: 21, coordinations: 0, adoptions: 0, messages: 0
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
