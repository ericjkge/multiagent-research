# indep_haiku_6_iso

## Setup

- protocol: open (Park et al. 2609.21032), shared log OFF (independent agents), share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-haiku-4-5-20251001
- effort: medium
- run budget: 36 of 36 used (2.65 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.280 (Claude Code)
- config fingerprint: aa8d14f41560

## Result

- rounds completed: 1
- training runs: 36 (2.65 GPU-h used)
- val_bpb: 1.012347 -> 0.999826
- improvement: 1.24%
- agent cost: $1.79
- wall clock: 2.66 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 4 | 1.001749 | 2 | 2 | 0 | 1 | $0.30 |
| a1 | 6 | 4 | 1.001943 | 2 | 1 | 0 | 1 | $0.29 |
| a2 | 6 | 5 | 0.999826 | 3 | 1 | 0 | 1 | $0.28 |
| a3 | 6 | 6 | 1.011649 | 2 | 1 | 0 | 1 | $0.34 |
| a4 | 6 | 3 | 1.001167 | 3 | 1 | 0 | 1 | $0.35 |
| a5 | 6 | 6 | 1.012817 | 1 | 1 | 0 | 1 | $0.23 |

Approaches claimed: a5: learning_rate_optimizer_tuning; a2: LR-and-depth-search; a4: depth-scaling; a0: optimizer_and_scheduling; a1: optimizer-tuning; a3: Optimizer & Learning Rate Tuning

Findings labelled weak: 0 of 13. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 36 | 8 | 0.999826 | a2/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.1706 |

## Errors

- crash rate: 22.2% (8/36)
- GPU spent on failed runs: 98s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 13, disconfirmations: 7, coordinations: 0, adoptions: 0, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 4 | 0 | $0.00 |
| a1 | 6 | 4 | 0 | $0.00 |
| a2 | 6 | 5 | 1 | $0.00 |
| a3 | 6 | 6 | 0 | $0.00 |
| a4 | 6 | 3 | 0 | $0.00 |
| a5 | 6 | 6 | 0 | $0.00 |
