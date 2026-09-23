# open_opus_6

## Setup

- protocol: open (Park et al. 2609.21032), shared log ON, share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-opus-5
- effort: medium
- run budget: 36 of 36 used (3.39 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: 4c07900ef647

## Result

- rounds completed: 1
- training runs: 36 (3.39 GPU-h used)
- val_bpb: 0.988763 -> 0.977896
- improvement: 1.10%
- agent cost: $35.93
- wall clock: 3.41 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 6 | 0.977973 | 4 | 2 | 3 | 1 | $6.07 |
| a1 | 6 | 6 | 0.982987 | 4 | 3 | 4 | 1 | $8.35 |
| a2 | 6 | 6 | 0.977896 | 2 | 4 | 5 | 1 | $6.26 |
| a3 | 6 | 6 | 0.988051 | 2 | 5 | 2 | 1 | $5.89 |
| a4 | 6 | 6 | 0.982903 | 3 | 4 | 2 | 1 | $5.19 |
| a5 | 6 | 6 | 0.978654 | 4 | 3 | 2 | 1 | $4.17 |

Approaches claimed: a3: batch-size & step-count economics; a0: batch-size / token-budget economics; a2: batch-size & optimizer-step economy; a4: batch-size & optimizer-step economy; a5: attention shape & block topology; a4: model shape & architecture scaling; a2: output head, embeddings & loss parameterization; a1: sequence-length curriculum & attention-cost scheduling; a0: optimizer & LR/WD schedule internals

Findings labelled weak: 2 of 19. Coordination notes: 8.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 36 | 0 | 0.977896 | a2/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 9 | 0.1829 |

## Errors

- crash rate: 0.0% (0/36)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 19, disconfirmations: 21, coordinations: 8, adoptions: 18, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 6 | 0 | $0.00 |
| a1 | 6 | 6 | 0 | $0.00 |
| a2 | 6 | 6 | 1 | $0.00 |
| a3 | 6 | 6 | 0 | $0.00 |
| a4 | 6 | 6 | 0 | $0.00 |
| a5 | 6 | 6 | 0 | $0.00 |
