# indep_opus_6

## Setup

- protocol: open (Park et al. 2609.21032), shared log OFF (independent agents), share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-opus-5
- effort: medium
- run budget: 36 of 36 used (3.30 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: c4115e7e3c57

## Result

- rounds completed: 1
- training runs: 36 (3.30 GPU-h used)
- val_bpb: 0.987981 -> 0.977649
- improvement: 1.05%
- agent cost: $37.20
- wall clock: 3.36 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 6 | 0.977649 | 3 | 3 | 0 | 1 | $10.76 |
| a1 | 6 | 6 | 0.978436 | 6 | 3 | 0 | 1 | $6.62 |
| a2 | 6 | 5 | 0.984440 | 3 | 4 | 0 | 1 | $6.96 |
| a3 | 6 | 6 | 0.982021 | 4 | 5 | 0 | 1 | $5.29 |
| a4 | 6 | 6 | 0.979960 | 4 | 2 | 0 | 1 | $3.89 |
| a5 | 6 | 6 | 0.984898 | 3 | 4 | 0 | 1 | $3.69 |

Approaches claimed: a3: batch-size / step-count allocation; a2: compute allocation & throughput; a5: batch-size economics (tokens/step vs number of steps); a0: residual-stream topology; a1: optimization scaling: batch size, step count, LR schedule shape; a4: compute allocation & shape at fixed FLOPs/token

Findings labelled weak: 5 of 23. Coordination notes: 1.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 36 | 1 | 0.977649 | a0/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.1628 |

## Errors

- crash rate: 2.8% (1/36)
- GPU spent on failed runs: 37s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 23, disconfirmations: 21, coordinations: 1, adoptions: 0, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 6 | 1 | $0.00 |
| a1 | 6 | 6 | 0 | $0.00 |
| a2 | 6 | 5 | 0 | $0.00 |
| a3 | 6 | 6 | 0 | $0.00 |
| a4 | 6 | 6 | 0 | $0.00 |
| a5 | 6 | 6 | 0 | $0.00 |
