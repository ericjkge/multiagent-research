# open_haiku_6

## Setup

- protocol: open (Park et al. 2609.21032), shared log ON, share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-haiku-4-5-20251001
- effort: medium
- run budget: 36 of 36 used (3.34 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.280 (Claude Code)
- config fingerprint: 78bdec04eee0

## Result

- rounds completed: 1
- training runs: 36 (3.34 GPU-h used)
- val_bpb: 0.996598 -> 0.995085
- improvement: 0.15%
- agent cost: $27.42
- wall clock: 3.52 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 6 | 0.995253 | 4 | 1 | 0 | 4 | $5.82 |
| a1 | 6 | 6 | 0.996159 | 4 | 3 | 0 | 1 | $3.38 |
| a2 | 6 | 6 | 0.995888 | 8 | 7 | 0 | 4 | $2.78 |
| a3 | 6 | 6 | 0.995457 | 2 | 4 | 0 | 1 | $0.78 |
| a4 | 6 | 6 | 0.997858 | 1 | 6 | 0 | 1 | $0.25 |
| a5 | 6 | 6 | 0.995085 | 7 | 1 | 0 | 1 | $14.40 |

Approaches claimed: a4: optimizer_tuning; a5: activation_tuning; a3: MLP-activation; a0: optimizer_hyperparams; a1: architecture_search; a2: optimizer_and_lr_tuning; a5: final_validation

Findings labelled weak: 0 of 26. Coordination notes: 1.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 33 | 0 | 0.995253 | a0/v0 |
| 0 | 36 | 0 | 0.995085 | a5/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 7 | 0.1001 |

## Errors

- crash rate: 0.0% (0/36)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 1, 'retries_that_recovered': 1, 'mean_attempts': 1.03}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 26, disconfirmations: 22, coordinations: 1, adoptions: 0, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 6 | 1 | $0.00 |
| a1 | 6 | 6 | 0 | $0.00 |
| a2 | 6 | 6 | 0 | $0.00 |
| a3 | 6 | 6 | 0 | $0.00 |
| a4 | 6 | 6 | 0 | $0.00 |
| a5 | 6 | 6 | 1 | $0.00 |
