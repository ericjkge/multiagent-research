# open_opus_6_iso

## Setup

- protocol: open (Park et al. 2609.21032), shared log ON, share per agent 6 runs
- agents: 6 x BoN=1 = 6 candidates/round
- models: claude-opus-5
- effort: medium
- run budget: 36 of 36 used (3.62 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.280 (Claude Code)
- config fingerprint: a1fbe00e821c

## Result

- rounds completed: 1
- training runs: 36 (3.62 GPU-h used)
- val_bpb: 1.011661 -> 0.984417
- improvement: 2.69%
- agent cost: $21.21
- wall clock: 3.64 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 6 | 6 | 0.986119 | 4 | 2 | 1 | 1 | $2.64 |
| a1 | 6 | 6 | 0.984417 | 3 | 2 | 0 | 1 | $2.21 |
| a2 | 6 | 6 | 0.986010 | 3 | 4 | 3 | 1 | $2.63 |
| a3 | 6 | 5 | 0.985510 | 3 | 3 | 3 | 1 | $6.97 |
| a4 | 6 | 6 | 0.986243 | 5 | 3 | 3 | 1 | $3.74 |
| a5 | 6 | 6 | 0.990240 | 4 | 4 | 3 | 1 | $3.03 |

Approaches claimed: a4: batch-size / step-count budget; a5: residual-stream topology (U-net skips); a2: attention shape & KV/value-embedding cost; a1: model shape & capacity allocation; a0: compute allocation (batch size x model shape x step count); a3: attention masking & data boundaries

Findings labelled weak: 5 of 22. Coordination notes: 3.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 36 | 1 | 0.984417 | a1/v0 |

## Idea diversity

| round | proposals | mean pairwise similarity |
|---|---|---|
| 0 | 6 | 0.1390 |

## Errors

- crash rate: 2.8% (1/36)
- GPU spent on failed runs: 900s
- repair: {'slots_that_retried': 0, 'retries_that_recovered': 0, 'mean_attempts': 1.0}

No crashed idea was echoed by another agent in the next round.

## Use of the shared directory

findings: 22, disconfirmations: 18, coordinations: 3, adoptions: 13, messages: 0
agents that published at least one finding: 6 of 6

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 6 | 6 | 0 | $0.00 |
| a1 | 6 | 6 | 1 | $0.00 |
| a2 | 6 | 6 | 0 | $0.00 |
| a3 | 6 | 5 | 0 | $0.00 |
| a4 | 6 | 6 | 0 | $0.00 |
| a5 | 6 | 6 | 0 | $0.00 |
