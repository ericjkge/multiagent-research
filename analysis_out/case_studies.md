# CORRECTION (Sep 23, 00:20): the "independent" cells were not independent

A reviewer of the Sep 22 snapshot found that agents in the independent control read each other's
work. An audit of the archived session transcripts (`python -m analysis.contamination results/indep_*`)
confirms it and shows it was systematic, not a one-off:

| cell | agents that read the shared score table | agents that inspected or checked out a peer's commit |
|---|---:|---:|
| `indep_opus_6` | 5 of 6 (up to 36 reads each) | 5 of 6 |
| `indep_opus_6_s2` | 5 of 6 | 5 of 6 |
| `indep_haiku_6` | 5 of 5 archived | 4 of 5 |
| `indep_haiku_6_s2` | 6 of 6 | 4 of 6 |
| `indep_sonnet_6`, `_s2` | no transcripts archived; same prompt and harness, assume the same |

Two harness defects caused it. The shared `results.tsv` (every agent's commit, score and run title) was
written into the run directory for all cells and the system prompt pointed every agent at it, sharing
on or off. And all agents worked in worktrees of one git repository, so a peer's commit could be
inspected, diffed or checked out with ordinary git commands (`git show`, `git diff`, `git reset --hard`);
verify only checked for recorded `arena-adopt` events. The winner of `indep_opus_6` (a0) built on a
peer's code after reading the table.

**What this withdraws.** Every statement below that the independent cells measure "no communication",
and every number derived from that contrast: section 4 ("the control answers the question"), section 6
(the seed comparison of the two arms), the "communication is worth at most 0.002" line in section 7 and
in the summary, and the same claim in the team messages of Sep 22. The measured scores of the independent
cells stand as observations of a third condition: passive visibility of peers' scores and code with no
coordination channel and no adoption protocol.

**What still stands.** Six agents beat one under both protocols on our box, and Eric's Sonnet ladder is
monotone on his; open beat rounds at every headcount on the same box (a workflow difference, not a
communication difference: session continuity, selection rule and attempt counts differ too); the rounds
protocol herds without any communication; the open cells' logs show real collaboration (18 adoptions,
19 findings, 21 disconfirmations in `open_opus_6`); model strength shows up as reliability of the search;
and none of this demonstrates recursive self-improvement, it studies the organisation of automated
research at fixed compute.

**Other reviewer points accepted.** Two seeds cannot bound an effect at "a couple of thousandths";
run-to-run noise (0.0008) is not the variability of a whole search; `open_opus_6_s2` ran one agent at
seven attempts and one at five after a budget rebuild; `indep_sonnet_6_s2` executed 37 attempts and the
37th is excluded from its record.

**Fix done, reruns done (section 9b).** True isolation for the independent arm: a per-agent score table, a separate git clone
per agent so peers' commits do not exist in its repository, a guard-hook denylist for the shared table,
peers' logs and worktrees, and a transcript scan that fails verification on any peer read. Reruns on Sep 23 (section 9b): no effect of sharing detected against a real no-access control; model strength is the larger lever.

---

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

## 4. The control answers the question (added 10:00) — WITHDRAWN, see the correction at the top

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

## 6. Second seeds of the headline pair (added 13:40) — WITHDRAWN as a communication comparison, see the correction

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
| Haiku, independent | | | 0.976806, 0.995414 (two seeds) |

Baseline 0.9973, run-to-run noise 0.0008.

**Headcount pays at six, not at three.** In every Opus family the six-agent cell beat the one-agent
cell by 0.003 to 0.006. Three agents did not beat one under the open protocol (0.9821 against 0.9807),
a difference inside the seed-to-seed spread of a cell (0.0015 to 0.0024 for six Opus agents). At this
budget the headcount effect is visible only at the top of the range.

**Communication is still not measurable.** (WITHDRAWN: the independent arm leaked, see the correction.) Six Opus agents with the shared directory: 0.97715 on average
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

**Sonnet with sharing, completed 16:35.** (The comparison with the independent seeds below is WITHDRAWN, see the correction.) `open_sonnet_6` finished at 0.985651, between the two
independent Sonnet seeds (0.9883 and 0.9770). Same verdict as for Opus: with cell-to-cell spread this
large, sharing does not move the number in a direction the data can resolve. The cell needed the
fresh-session fallback for four of its six agents and crashed 8 percent of runs; its agents did adopt
each other's results (the winning run is a2's "DEPTH=10 on top of adopted batch 2^18 + softcap 10 base",
a depth change stacked on a peer's adopted stack), so the protocol worked once the sessions did.

**The Haiku replicate, completed 17:47, corrects the surprise.** The second seed of the Haiku control
finished at 0.995414: 36 runs, no crashes, and not one agent left the learning-rate knobs
(`results/indep_haiku_6_s2/winner.diff` is two learning rates). The two Haiku seeds are 0.019 apart,
the widest spread of any arm, against 0.002 for the two independent Opus seeds. So the honest model-strength
statement is not "Haiku matches Opus" but: a weak model's cell is a lottery on whether one of its six
agents happens to try the structural changes (width, batch, window), while Opus agents find them in
every seed. Model strength buys reliability of the search, not a different optimum. For the takeoff
argument that is the more useful finding: at fixed compute, a stronger researcher does not find better
ideas here, it stops wasting runs on ideas that cannot pay.

Final standing, 14 cells, all at full budget (opus_3 and opus_6 at 34 by protocol). Nothing is still running.

## 8. The teammates' Sonnet rounds family (added Sep 22, 23:55)

Eric pushed three round-protocol Sonnet cells run on his own box: `sonnet_1` (best-of-5), `sonnet_3`
(best-of-2) and `sonnet_6`, at 32, 32 and 33 runs (the same partial-round limit as our `opus_3` and
`opus_6`). All three pass verify. A stopped pilot of `sonnet_6` is archived beside them and excluded
from every table.

**Read gains, not finals, across boxes.** His box measured the untouched baseline at 1.0140 against our
0.9973, a gap twenty times the run-to-run noise, so absolute finals are not comparable between the two
boxes. Within his box the ladder is clean and monotone:

| cell (Eric's box, baseline 1.0140) | runs | final | gain |
|---|---:|---:|---:|
| `sonnet_1`, best-of-5 | 32 | 1.0023 | 0.0117 |
| `sonnet_3`, best-of-2 | 32 | 0.9930 | 0.0211 |
| `sonnet_6` | 33 | 0.9780 | 0.0360 |

Six beat three beat one, and the six-agent gain is the largest of any cell in the study. On our box the
Sonnet cells gained 0.0117 (`open_sonnet_6`), 0.0090 and 0.0203 (the two independent seeds): Sonnet's
cell-to-cell spread is wide enough that a single-seed ladder can look either monotone or flat. What his
ladder does add is a second, independent confirmation of the headcount effect (six over one, here by
0.024), on a different box, with a different protocol.

**Do not read protocol from this.** `sonnet_6` (rounds, his box) at 0.0360 against `open_sonnet_6`
(open, our box) at 0.0117 is a cross-box, single-seed comparison and says nothing about rounds versus
open. The within-box protocol comparison remains the Opus grid: open beat rounds at one, three and six
agents.

## 9. The isolated reruns (Sep 23, 00:42 to 07:12)

Three independent cells rerun on a fresh box (pod 4) under the repaired isolation: private one-commit
clones, per-agent score tables, the guard-hook denylist, and a transcript scan at archive time. Both
finished cells scan clean: every agent, zero reads of the shared table, zero reads of peers' logs, zero
references to peers' commits (`results/<cell>/isolation.txt`).

The box is about 20 percent slower than Monday's (same GPU model and clocks, 387M tokens in the
five-minute baseline against 486M), so its untouched baseline is 1.0123 instead of 0.9973 and its finals
must not be set beside Monday's. The open arm is therefore being rerun on this box as well
(`open_opus_6_iso`), so that the sharing versus no-sharing comparison is within one machine.

| cell (pod 4, baseline 1.0123) | runs | final | gain | isolation |
|---|---:|---:|---:|---|
| `indep_opus_6_iso` | 36 | 0.982146 | 0.0302 | clean, 6 of 6 agents |
| `indep_sonnet_6_iso` | 36 (7 crashes) | 0.996232 | 0.0161 | clean, 6 of 6 agents |
| `open_opus_6_iso` | running | | | sharing on by design |
| `indep_haiku_6_iso` | running | | | |

Two things are already visible. Truly isolated Opus agents get a long way on their own: a 0.030 gain,
larger than any Monday cell's, though on a slower box where the same knobs are worth more. And the
Sonnet gap to Opus (0.014) is of the same order as Monday's, with the Sonnet crash rate again far higher
(7 of 36 against 0 of 36). The communication comparison waits for `open_opus_6_iso`.

### 9b. All four isolated cells in (Sep 23, 07:12 finish; written 11:45)

| cell (pod 4, baseline 1.0123) | runs | crashes | final | gain | isolation scan |
|---|---:|---:|---:|---:|---|
| `open_opus_6_iso` (sharing on) | 36 | 1 | 0.984417 | 0.0279 | not required |
| `indep_opus_6_iso` (no access to peers) | 36 | 0 | 0.982146 | 0.0302 | clean, 6 of 6 |
| `indep_sonnet_6_iso` | 36 | 7 | 0.996232 | 0.0161 | clean, 6 of 6 |
| `indep_haiku_6_iso` | 36 | 8 | 0.999826 | 0.0125 | clean, 6 of 6 |

**Sharing versus true isolation, one machine, one seed each.** Six Opus agents that could not see each
other at all finished at 0.9821; six that shared everything through the directory finished at 0.9844.
The difference, 0.0023 in favour of the isolated agents, is inside the spread measured between seeds
of one arm on Monday (0.0015 to 0.0024), and a one-seed-per-arm design detects nothing smaller than
about 0.006. So the statement the reviewer asked for now has its control: at 36 runs of five minutes,
no effect of sharing was detected, in either direction. The open cell still shows the protocol working
(adoptions, findings, disconfirmations on its log); it did not show a better number.

**Model strength, now measured properly.** With real isolation the ladder is unambiguous: Opus gained
0.030, Sonnet 0.016, Haiku 0.013, and the weak models crashed a fifth of their runs. Monday's
"Haiku matches Opus" was copying; alone, Haiku does not find the width, batch and window changes that
carry the Opus result. Model strength is worth about twice as much as headcount here (six Opus over
one Opus was 0.003 to 0.006 on Monday's box), and far more than sharing.

**Caveats that stay.** One seed per arm on this box; the box is slower than Monday's, so its gains are
larger for the same changes and are compared only within it; the Monday `indep_*` cells remain
contaminated and are kept only as the record of that.
