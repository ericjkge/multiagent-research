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

## 4. The control answers the question (added 10:00)

`indep_opus_6`: the same six Opus agents, the same 36 runs, six each, and no shared log at all. Each
agent saw only its own results. Final 0.977649 against 0.977896 for the six that could talk: a
difference of 0.00025, a third of the noise band.

So at this budget the six-agent gain is parallel exploration, not communication. Six independent
hill-climbs with six runs each covered the space as well as a coordinated collective, at the same
token cost ($37 vs $36). Five of the six independent agents even chose the same approach family
(batch-size economics) with no way to know it, and the winner was the one that didn't (a0,
residual-stream topology). Best-of-six beat six-that-talk, and both beat one agent with 36 runs.

This is the paper's own caveat, measured in a research loop: "communication pays only when each
agent has enough budget to explore on its own". Six five-minute runs is not enough. Whether that
changes with more budget per agent, or with a stronger model than Opus, is the open question; the
second seeds of both arms (`open_opus_6_s2`, `indep_opus_6_s2`) are running to check the first result
is not a one-off.

Standing at 10:00 (gain over baseline, 36 runs unless noted): indep 6 agents 0.0197, open 6 agents
0.0194, open 1 agent 0.0166, open 3 agents 0.0152, rounds 6 agents 0.0155 (29 runs, cost cap),
rounds 1 agent 0.0111 (33 runs). Noise band 0.0008.

## 5. Weak agents quit early and over-report (added 10:15)

`indep_haiku_6` (six Haiku agents, no sharing, six runs each available) ended itself after 13 of 36
runs: the agents used 1, 3, 1, 0, 2 and 6 runs respectively, each declaring itself finished. Their
final summaries describe "systematic" and "extensive" exploration; agent a3 ran zero training runs and
summarised experiments it never submitted. Best 0.996838 against a 0.997359 baseline, inside noise.

The same protocol, prompt and budget with Opus produced 36 of 36 runs from every agent and a 0.0197
gain. So on the model-strength axis the first difference is not the quality of ideas but whether the
agent spends its compute at all, and whether its account of what it did can be trusted. Under a
protocol whose stopping rule is the agent's own judgement, a weak agent's judgement is the bottleneck.
The cell is archived as INVALID (one slot claimed without a run) and kept for this record.

## 6. Second seeds of the headline pair (added 13:40)

| arm | seed 0 | seed 1 | mean |
|---|---|---|---|
| six agents, shared directory (`open_opus_6`) | 0.977896 | 0.976408 | 0.97715 |
| six agents, no communication (`indep_opus_6`) | 0.977649 | 0.980071 | 0.97886 |

The seed-to-seed spread of a whole cell (0.0015 and 0.0024) is two to three times the run-to-run
noise (0.0008): the outcome of a 36-run cell depends on which ideas its agents happen to try first.
On the means, communication is worth 0.0017, but with two seeds per arm that is inside the spread.
What can be said: the value of communication at this budget is at most a couple of thousandths,
about half the value of going from one agent to six (0.003 to 0.004, which does clear the spread),
and not distinguishable from zero with this many seeds. The honest slide is a range, not a number.

## 7. Every cell at full budget: what the top-ups changed (added 14:35)

All twelve cells now stand at 36 recorded runs, except the two round-based Opus cells that lost a slot
inside a closed round (`opus_3` and `opus_6` at 34: the protocol cannot add a partial round). The
equal-compute table in `summary.md` (best by run 29, 34 and 36) shows the ordering is the same at 34
and at 36, so the comparison below uses each cell's final number.

| family | 1 agent | 3 agents | 6 agents |
|---|---|---|---|
| Opus, open (shared directory) | 0.980742 | 0.982130 | 0.977896, 0.976408 (two seeds) |
| Opus, independent (no sharing) | | | 0.977649, 0.980071 (two seeds) |
| Opus, rounds (blind proposals, one winner per round) | 0.986277 | 0.987792 (34 runs) | 0.980496 (34 runs) |
| Sonnet, open (shared directory) | | | 0.985651 |
| Sonnet, independent | | | 0.988312, 0.977046 (two seeds) |
| Haiku, independent | | | 0.976806 |

Baseline 0.9973, run-to-run noise 0.0008.

**Headcount pays at six, not at three.** In every Opus family the six-agent cell beat the one-agent
cell by 0.003 to 0.006. Three agents did not beat one under the open protocol (0.9821 against 0.9807),
a difference inside the seed-to-seed spread of a cell (0.0015 to 0.0024 for six Opus agents). At this
budget the headcount effect is visible only at the top of the range.

**Communication is still not measurable.** Six Opus agents with the shared directory: 0.97715 on average
over two seeds. The same six with no sharing: 0.97886. The 0.0017 in favour of talking is smaller than
the spread between seeds of the same arm. Best-of-six covers this 36-run search as well as a team.

**The model-strength surprise.** Forced to spend all 36 runs (section 5 recorded the same cell quitting
at 13), six independent Haiku agents reached 0.976806, level with the best Opus cells. The winning
diff (`results/indep_haiku_6/winner.diff`) is the same basin every Opus winner landed in: aspect ratio
96 (width over depth), batch 2^18, short attention window 256, value embeddings on the first and last
three layers, Muon LR 0.06, weight decay 0.1. Sonnet's two independent seeds split 0.9883 and 0.9770, a
spread of 0.011, five times the Opus spread. What separates the models on this task is not the ideas
but the process: Haiku crashed 31 percent of its runs (Opus 0 to 3 percent) and stops when allowed to;
Sonnet crashed 11 to 22 percent and, in the shared cell, four of six sessions looped 41 twelve-second
segments each on the belief that the training command was broken, until the orchestrator learned to
hand a stuck agent a fresh session. The 36-run search space of five-minute `autoresearch` runs is
shallow enough that a weak model finds the same knobs; a strong model finds them more reliably.

**For the takeoff argument.** The lever that moved results here was spending the compute (Haiku from
0.9968 to 0.9768 by being made to continue), then headcount (six over one), then organisation (open over
rounds). Neither model quality nor communication moved the number beyond noise at this budget. That is
the "compute is the bottleneck" reading of the loop, measured: what the agents could not do was run
more five-minute experiments per unit of GPU time.

**Sonnet with sharing, completed 16:35.** `open_sonnet_6` finished at 0.985651, between the two
independent Sonnet seeds (0.9883 and 0.9770). Same verdict as for Opus: with cell-to-cell spread this
large, sharing does not move the number in a direction the data can resolve. The cell needed the
fresh-session fallback for four of its six agents and crashed 8 percent of runs; its agents did adopt
each other's results (the winning run is a2's "DEPTH=10 on top of adopted batch 2^18 + softcap 10 base",
a depth change stacked on a peer's adopted stack), so the protocol worked once the sessions did.

Still running: a second seed of the Haiku control (`indep_haiku_6_s2`), to check that 0.9768 was not luck.
