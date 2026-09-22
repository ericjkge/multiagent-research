# open_opus_6_s2

## Setup

- protocol: open (Park et al. 2609.21032), shared log ON, share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-opus-5
- effort: medium
- run budget: 36 of 36 used (3.41 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.278 (Claude Code)
- config fingerprint: bfb3a851e58c

## Result

- rounds completed: 1
- training runs: 36 (3.41 GPU-h used)
- val_bpb: 0.987984 -> 0.976408
- improvement: 1.17%
- agent cost: $66.10
- wall clock: 5.00 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 6 | 0.976408 | 5 | 1 | 3 | 1 | $12.23 |
| a1 | 6 | 6 | 0.980292 | 4 | 4 | 4 | 1 | $10.37 |
| a2 | 6 | 6 | 0.977161 | 4 | 2 | 2 | 1 | $11.04 |
| a3 | 5 | 5 | 0.978308 | 4 | 4 | 4 | 0 | $8.49 |
| a4 | 6 | 6 | 0.979464 | 3 | 3 | 3 | 1 | $14.13 |
| a5 | 7 | 7 | 0.976458 | 3 | 4 | 4 | 1 | $9.84 |

Approaches claimed: a4: batch-size & step-count economics; a1: batch-size & update-count economics; a5: batch-size / step-count economics; a2: batch-and-sequence economics; a0: residual topology & layer connectivity; a2: model shape and attention structure; a5: model shape & capacity scaling; a1: model shape & capacity scaling; a5: optimizer & LR-schedule surface (final claim for a5); a3: throughput engineering (tokens/sec at fixed dynamics); a3: attention internals (rotary base, softmax scale, window size, head gating); a2: attention structure at the known frontier (a2, continuing slot #9)

Findings labelled weak: 5 of 23. Coordination notes: 6.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 35 | 0 | 0.976408 | a0/v0 |
| 0 | 36 | 0 | 0.976408 | a0/v0 |
| 0 | 36 | 0 | 0.976408 | a0/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 12 | 0.1393 |

## Errors

- crash rate: 0.0% (0/36)
- GPU spent on failed runs: 0s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 23, disconfirmations: 18, coordinations: 6, adoptions: 20, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 6 | 3 | $0.00 |
| a1 | 6 | 6 | 0 | $0.00 |
| a2 | 6 | 6 | 0 | $0.00 |
| a3 | 5 | 5 | 0 | $0.00 |
| a4 | 6 | 6 | 0 | $0.00 |
| a5 | 7 | 7 | 0 | $0.00 |
