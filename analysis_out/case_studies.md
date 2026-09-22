# Case studies from the first four Opus cells (Sep 22, 06:50)

Entry numbers (`#n`) refer to `results/<cell>/log.md`. Baseline 0.9973, noise band 0.0008 (six identical runs).

## 1. Six open agents: the protocol did what the paper says it does

**Collision, then self-correction (first ten minutes).** Four of the six agents claimed the same approach
family, batch-size economics, within a minute of each other (#2 to #5). Agent a5 called it out (#7:
"4 of 6 agents on one axis; suggest at least two of you pivot"). Three of them withdrew and re-claimed
different families on the record: a4 to model shape (#8), a2 to output head and loss (#10), a0 to
optimizer schedules (#13). a3 kept the batch family and announced its plan (#17). No orchestrator
intervened; the coordination channel and the "distinct approach" rule did it.

**A lineage built by five different agents.** Every adoption in the cell cites a measured number
that beat the adopter's own best, as the rule requires:

| step | who found it | what | score | adopted by |
|---|---|---|---|---|
| 1 | a3 | batch 2^19 to 2^18 | 0.988763 | all five others (#20, #23, #26, #29, #33) |
| 2 | a5 | sliding window 256 on top of it | 0.987049 | a4, a2, a3 (#40, #45, #50) |
| 3 | a4 | shallow and wide: depth 6, dim 640 | 0.982903 | a0, a2, a1, a5 (#57 to #68) |
| 4 | a0 | warmdown ratio 1.0 on that | 0.982265 | a2, a1, a3 (#75 to #81) |
| 5 | a5 | width to dim 768 | 0.978654 | a0, a2, a1 (#89 to #96) |

The winning stack has five components from four agents. Nobody could have run this lineage alone in
36 runs: each step was found by an agent working its own family while the others explored elsewhere.

**Negative results were published, with mechanisms.** 21 disconfirmations, e.g. #22 "depth 8 to 10 is
worse: tokens seen fell 472M to 299M", #65 "2^17 loses for a real reason, not an overhead artifact",
#80 "tail weight averaging hurts, measured within a single run as a clean A/B". Two agents
independently found the width ladder turns over at dim 768 (#102, #116). a5 turned an accidental
duplicate into a replicate and published a run-to-run noise estimate (#61).

**Cost:** $36 of notional agent spend for 36 runs, six agents.

## 2. Six agents in lockstep rounds: herding without any communication

The round protocol keeps proposals blind: agents propose without seeing each other. They converged anyway.

| round | what the six proposed | outcome |
|---|---|---|
| 0 | five of six: "halve the batch 2^19 to 2^18" (a3: U-net skips) | a5's copy won, 0.988076 |
| 1 | all six: HEAD_DIM 128 to 64, some plus depth 12 | nothing beat the baseline |
| 2 | all six again: HEAD_DIM 64 "the unrun confirmed win" | nothing beat the baseline |
| 3 | all six: warmdown 0.5 to 0.7 | 0.987227 |
| 4 | three: shorter window; two: device batch 128; one: RoPE base | 0.981817 |

Twelve runs, rounds 1 and 2, went to the same idea six times over, because the lineage rule
advances only one winner per round and the agents kept re-proposing what they believed was a
"confirmed win" that had never made it into the shared baseline. Same model, same compute, no talking:
identical priors produce identical proposals. The cell stopped at 29 runs when it hit its cost
ceiling, having spent $291 in notional agent calls, eight times the open cell, for a worse score.

## 3. One agent under each protocol

`open_opus_1` (one long session, memory of all its own results): 0.980742, improving until run 36,
and a final summary attributing the gain to six stacked changes with per-change deltas. `opus_1`
(fresh session each round, best-of-5): 0.986277, all gains in rounds 2 and 3, then three empty rounds.
Open beat rounds by 0.0055 at a tenth of the agent cost.

## What this says, so far

At 36 runs with Opus, copies help (six beat one under both protocols) and organisation matters as much
as headcount (open beat rounds at both sizes). Whether the six-agent gain is communication or just six
parallel attempts is what `indep_opus_6` (same six agents, no shared log) is measuring now.
