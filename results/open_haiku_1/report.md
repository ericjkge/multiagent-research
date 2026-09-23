# open_haiku_1

## Setup

- protocol: open (Park et al. 2609.21032), shared log ON, share per agent 36 runs
- agents: 1 x BoN=1 = 1 candidates/round
- models: claude-haiku-4-5-20251001
- effort: medium
- run budget: 36 of 36 used (3.33 GPU-h)
- autoresearch commit: 228791f
- claude: 2.1.280 (Claude Code)
- config fingerprint: 78b7e30b2af3

## Result

- rounds completed: 1
- training runs: 36 (3.33 GPU-h used)
- val_bpb: 0.996598 -> 0.992818
- improvement: 0.38%
- agent cost: $0.83
- wall clock: 5.42 h
- stopped because: all agents finished

## Open protocol activity

| agent | runs | ok | best val_bpb | findings | disconfirm. | adoptions | segments | cost |
|---|---|---|---|---|---|---|---|---|
| a0 | 36 | 36 | 0.992818 | 3 | 0 | 0 | 1 | $0.83 |

Approaches claimed: a0: optimizer_tuning

Findings labelled weak: 0 of 3. Coordination notes: 0.

## Progress per round

| round | candidates | crashes | best val_bpb | winner |
|---|---|---|---|---|
| 0 | 36 | 0 | 0.992818 | a0/v0 |

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

findings: 3, disconfirmations: 0, coordinations: 0, adoptions: 0, messages: 0
agents that published at least one finding: 1 of 1

## Per-agent attribution

| agent | runs | successful | rounds won | cost |
|---|---|---|---|---|
| a0 | 36 | 36 | 1 | $0.00 |
