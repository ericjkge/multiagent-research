# indep_opus_6_iso

## Setup

- protocol: open (Park et al. 2609.21032), shared log OFF (independent agents), share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-opus-5
- effort: medium
- run budget: 36 of 36 used (3.41 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.280 (Claude Code)
- config fingerprint: 04d5cda0820b

## Result

- rounds completed: 1
- training runs: 36 (3.41 GPU-h used)
- val_bpb: 1.001379 -> 0.982146
- improvement: 1.92%
- agent cost: $17.11
- wall clock: 3.47 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 6 | 0.986791 | 5 | 2 | 0 | 1 | $1.91 |
| a1 | 6 | 6 | 0.994308 | 4 | 3 | 0 | 1 | $3.29 |
| a2 | 6 | 6 | 1.005200 | 3 | 3 | 0 | 1 | $3.74 |
| a3 | 6 | 6 | 0.988145 | 4 | 3 | 0 | 1 | $2.02 |
| a4 | 6 | 6 | 0.982146 | 5 | 2 | 0 | 1 | $2.30 |
| a5 | 6 | 6 | 0.988531 | 3 | 4 | 0 | 1 | $3.84 |

Approaches claimed: a0: batch-size / update-count economics; a1: optimization scale (batch size x step count x LR schedule); a3: compute allocation: batch size x model shape; a4: compute allocation: batch-size/step-count economics and model shape (depth, width, MLP ratio); a2: compute-optimal model scaling; a5: compute allocation: model shape vs optimizer steps

Findings labelled weak: 3 of 24. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 24 | 0 | 0.982146 | a4/v0 |
| 0 | 36 | 0 | 0.982146 | a4/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.2397 |

## Errors

- crash rate: 0.0% (0/36)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 24, disconfirmations: 17, coordinations: 0, adoptions: 0, messages: 0
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
