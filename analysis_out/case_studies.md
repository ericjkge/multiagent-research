# Research behavior and result interpretation — Sep 23, 2026

This replaces the evolving provisional interpretation with an account of the archived results at
`53492bf`. Earlier text remains in Git history. Scores below are best observed validation bits per
byte (lower is better). A **search** is a full agent configuration; a **training attempt** is one
experiment within it. Most configurations have only one complete search. Observations, agent
explanations and causal conclusions must not be treated as interchangeable.

## Control validity: original searches versus repaired reruns

The six original `indep_{haiku,sonnet,opus}_6` and `_s2` searches used a harness that exposed the
shared score table and peer Git objects. They cannot serve as no-communication controls. The
archived audit directly documented peer access in the Opus and Haiku searches; its Sonnet coverage
was insufficient to demonstrate isolation. This is not evidence that every Sonnet agent copied.
The winning agent in the original `indep_opus_6` inspected and used peer code.

The original comparison and its derived claims about the value of communication are withdrawn.
Their scores remain recorded observations under unintended information access, not a controlled
measurement of passive sharing either: access and use were not standardized across agents.

The three `_iso` independent reruns use separate clones, private score tables, access checks and
archived transcript scans. All six agents in each scan report no detected shared-table reads,
peer-log reads or peer-commit references. This is evidence that the known leak was addressed;
a pattern-based scan is not proof against every possible information channel.
The name `open_opus_6_iso` denotes the corresponding rerun with sharing **on**.

Sources: [contamination scanner](../analysis/contamination.py), isolation reports for
[Opus](../results/indep_opus_6_iso/isolation.txt),
[Sonnet](../results/indep_sonnet_6_iso/isolation.txt), and
[Haiku](../results/indep_haiku_6_iso/isolation.txt).

## 1. Open Opus: agents redirected overlapping proposals

In `open_opus_6`, four agents initially claimed batch-size research. Agent a5 flagged the overlap
in entry #7. Agents a4, a2 and a0 then withdrew and announced other directions (#8, #10, #13):
model shape, output/embedding changes, and optimizer schedules. This is an observed response to
communication. It does not establish how much performance the reallocation caused.

## 2. Open Opus: improvements passed between agents

The recorded findings include a3's smaller batch (0.988763), a5's shorter window on that base
(0.987049), a4's depth-6/width-640 model (0.982903), a0's longer decay on that shape (0.982265),
and a5's width-768 model (0.978654). These are related branches, not a single sequential path:
a5 widened a4's base, and later runs combined the wider model with a0's schedule.

The log contains 18 adoption events, 19 findings and 21 disconfirmations. The final winner was
0.977896. Its author, a2, explicitly noted that its own added embedding-placement change differed
from a0's 0.977973 control by only 0.000077 and did not merit credit (#113).

The record supports cumulative collaboration and careful attribution. It does not show that a
single agent could not have discovered the same changes, or that every inherited component had a
separately established causal benefit. Agent estimates of noise and mechanisms are hypotheses or
local measurements, not a general uncertainty model for the study.

Source for sections 1–2: [open Opus research log](../results/open_opus_6/log.md),
[machine-readable entries](../results/open_opus_6/log.jsonl).

## 3. Rounds: discussion did not reliably prevent duplicate work

Initial proposals can be blind, but the rounds workflow is **not** a no-communication condition.
The six-agent Opus archive contains extensive messages and responses. Agents repeatedly noticed
similar proposals and tried to avoid duplication. In #48, a5 criticized repeated clustering;
in #67, a0 reported that agents had over-corrected and several final files were identical anyway.

This shows a gap between describing a coordination problem and consistently solving it. Proposal
similarity is not the same as executed-code duplication; exact duplication claims require comparing
the archived code, not just counting similar proposal titles. The record does not establish that
communication caused herding or that the lineage-selection rule was its sole cause.

Source: [rounds Opus entries #48 and #67](../results/opus_6/log.jsonl).

## 4. Agents audited measurement claims and revised explanations

In rounds Opus, a2 proposed head dimension 160 (#192). Other agents inspected its archived artifacts
and reported that the supposed measurement reused earlier logs and another agent's run identity
(#206, #208, #214). They also inspected the installed FlashAttention package from the Hugging Face
cache and reported that dimension 160 maps to a 192-wide kernel, complicating the computational-cost
argument. These are documented investigative behaviors; the artifact problem prevents treating that
candidate as a fresh measurement of the proposed architecture.

In open Opus, a1 initially suspected a data-loader bottleneck (#32), then revised its explanation
after the faster-loader experiment produced little improvement (#47). This is evidence of a public
correction, not proof that every subsequent performance explanation was correct.

Sources: [rounds investigation](../results/opus_6/log.jsonl),
[open findings](../results/open_opus_6/log.jsonl).

## 5. Agent count: observed effects depend on configuration

| Setup and workflow | 1 agent | 3 agents | 6 agents | Actual attempts per search |
|---|---:|---:|---:|---|
| Opus, open (baseline 0.9973) | 0.980742 | 0.982130 | 0.977896; repeat 0.976408 | 36 each |
| Opus, rounds (baseline 0.9973) | 0.986277 | 0.987792 | 0.980496 | 36 / 34 / 34 |
| Sonnet, rounds (baseline 1.0140) | 1.002279 | 0.992950 | 0.978041 | 32 / 32 / 33 |
| Sonnet, open (baseline 1.0125) | 0.979324 | 0.993862 | not run on this setup | 36 / 36 |
| Haiku, open (baseline 0.9966) | 0.992818 | 0.990882 | 0.995085 | 36 each |
| Haiku, rounds (baseline 0.9966) | 0.994120 | 0.993742 | 0.994321 | 20 / 21 / 28 |

Compare within rows, not across hardware setups. Six beat one in the original Opus families and
Sonnet rounds. Three beat one in Sonnet rounds and Haiku open. Haiku open six did worst; Sonnet open
one beat three. The Haiku rounds family has unequal realized budgets. These observations do not
establish a monotonic scaling law, a headcount threshold, or that additional agents never help Haiku.

At a fixed total budget, agent count changes both the number of research trajectories and the budget
available to each. Search depth versus breadth is a plausible explanation, not a measured mechanism.
The six agents within a team are not six independent replications of the team treatment.

Source: [archived summaries](summary.md); hardware groups follow the recorded experiment context,
not an assumption that similar baseline scores make machines equivalent.

## 6. Workflow: the original Opus comparison favors open sessions

Open finished better than rounds at 1, 3 and 6 agents in the original Opus experiments. Rounds also
change memory/session continuity, candidate selection and when results become available. Actual
attempt counts differ, and solo Opus rounds was resumed with a smaller final batch of candidates.
This is an observed workflow difference, not an isolated effect of communication or scheduling.

It is not uniform across models: Haiku open six finished worse than Haiku rounds six, though the
latter used only 28 attempts. Cross-machine Sonnet open-versus-rounds comparisons are confounded.
Reported agent-dollar totals are not equivalent to measured inference tokens or verified invoices;
no claim of an eightfold token advantage follows from the summary's dollar column alone.

## 7. Execution reliability and accounting constrain interpretation

The archive contains 29 finished search summaries and a separate stopped Sonnet pilot. Six original
independent searches are excluded as controls. A verifier's OK status is not a scientific-validity
certificate, particularly when it tolerates known deviations.

The Haiku rounds logs each contain 36 candidate records, but the GPU timelines contain only
20, 21 and 28 attempts. Those timelines show respectively 0, 2 and 5 attempts without a metric;
the summary's candidate-based crash percentages measure something different. Candidate records
without corresponding GPU execution must not be counted as completed training attempts.

Other documented deviations include seven/five attempts for two agents in `open_opus_6_s2`, and
37 actual attempts in `indep_sonnet_6_s2`, with one later excluded from its main record. Search
resumes, session resets and budget adjustments also occurred. These differences matter for cost,
wall time and reproducibility even where they do not change the reported best score.

The open Haiku six-agent log records repeated early stopping and forced continuation, including a
fresh session after three segments produced no training. This supports a concrete observation about
model–harness reliability. It does not by itself distinguish misunderstanding, tool problems,
context problems and deliberate stopping.

Sources: per-search `gpu_timeline.jsonl`, `provenance.json` and
[open Haiku notes](../results/open_haiku_6/log.jsonl). Existing progress-table limitations are
explicitly identified in [summary.md](summary.md).

## 8. Scope of uncertainty and cross-machine comparisons

The historical baseline measurements differ: approximately 0.9973 on the original pods, 1.012347 on
pod 4, 1.012531 for the new Sonnet open pair, 1.014027 for Sonnet rounds, and 0.996598 for the Haiku
family. Equal GPU model names or similar baseline losses do not establish equivalent throughput or
response to a training-code change. Subtracting baseline loss does not remove that confound.

A training-run noise estimate is not the variability of a full adaptive research search. The two
original open Opus six-agent repeats differ by 0.001488, but two observations do not establish a
stable variance, a confidence interval or a minimum detectable effect on a different setup. Config
labels such as `_s2` denote repeat searches; they do not establish independently controlled training
seeds. Archived training code uses fixed seeds in many runs.

Best scores were selected using the same research validation metric queried during search. This
analysis does not establish independently replicated winner performance or held-out generalization.
Equal attempt ceilings are not equal realized GPU time, successful measurements or total AI compute.

## 9. Repaired independent reruns

The three independent `_iso` searches and the communicating Opus counterpart were completed on pod 4.
They use the same recorded substrate commit, model effort and baseline, and 36 attempts each.
The independent logs show no detected use of the known peer-information channels. Original
contaminated scores are not pooled with these reruns.

### 9b. Communication and model comparisons on pod 4

| Configuration, six agents | Attempts | Without metric | Best score | Gain from baseline 1.012347 |
|---|---:|---:|---:|---:|
| Opus, communicating (`open_opus_6_iso`) | 36 | 1 | 0.984417 | 0.027930 |
| Opus, independent (`indep_opus_6_iso`) | 36 | 0 | 0.982146 | 0.030201 |
| Sonnet, independent (`indep_sonnet_6_iso`) | 36 | 7 | 0.996232 | 0.016115 |
| Haiku, independent (`indep_haiku_6_iso`) | 36 | 8 | 0.999826 | 0.012521 |

**Observation: communication did not outperform isolation in this pair.** Independent Opus finished
0.002271 lower. There is one complete search per condition. This does not establish that communication
hurts, is equivalent to independent search, or has a benefit below a specified bound. No general
0.006 detection threshold is justified, and the proposed explanation that six attempts per agent
is too little for communication to pay remains untested.

**Exploratory mechanism: reuse versus the best discovery.** For each agent, take its minimum valid
candidate score, then take the median across the six agents. That gives 0.9860645 for communicating
Opus and 0.988338 for independent Opus. The team had a better middle-performing agent but a worse
best result. Its log records 13 adoptions. This is consistent with sharing spreading useful results
without improving the best discovery in this search; it is not evidence of a general causal effect.
Agent-level outcomes within the communicating team are dependent and cannot be treated as six
independent treatment replicates.

**Observation: Opus did better in the independent model comparison.** It found the best score and
had no failed measurements, versus seven for Sonnet and eight for Haiku. This compares each model
operating through this harness, including implementation and recovery, not idea quality alone.
One search per model cannot establish population reliability or rank this effect against headcount
and workflow effects measured elsewhere.

The winning code changes differ. Independent Opus used depth 8 / width 768 and batch 2^17;
communicating Opus used depth 12 / width 512, batch 2^18 and a shorter window. Independent Sonnet
increased width to 640; independent Haiku increased depth to 10 and embedding learning rate to 0.7.
Thus the claim that every winner found the same width/batch/window configuration is incorrect.
These selected configurations do not establish unique optima or isolate each component's effect.

Sources: logs for [independent Opus](../results/indep_opus_6_iso/log.jsonl) and
[communicating Opus](../results/open_opus_6_iso/log.jsonl); winning diffs for
[independent Opus](../results/indep_opus_6_iso/winner.diff),
[communicating Opus](../results/open_opus_6_iso/winner.diff),
[Sonnet](../results/indep_sonnet_6_iso/winner.diff), and
[Haiku](../results/indep_haiku_6_iso/winner.diff).

## 10. Research conclusion and RSI relevance

The archive demonstrates agents communicating about experiments, reusing peers' work, revising
claims, and sometimes discovering problems in measurement records. It also documents persistent
coordination failures, early stopping, failed execution and a compromised original control.

The repaired communication comparison did not favor sharing in its one matched pair. Agent-count
results are mixed, and the repaired model comparison favors Opus with fewer failed measurements.
These are exploratory findings about the organization of automated ML research. They do not show
recursive improvement of the researcher models, an RSI takeoff rate, a general scaling law, or a
reliable performance multiplier from communication.
