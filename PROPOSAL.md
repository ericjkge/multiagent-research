# Can AI agents collaborate on ML research?

CS 2881R in-class experiment, Thursday September 24, 2026 (Recursive Self-Improvement and AI
Trajectories, guests Dwarkesh Patel and Daniel Kokotajlo). Team: Anthony Shen, Eric Ge, Riddhi
Bhagwat, Alvin Ekelund. Proposal v2, September 21, 2026.

## The RSI question this measures

Recursive self-improvement, in the sense the forecasts use, is AI doing the AI research that makes the
next AI better. Whether that loop runs away or crawls depends on one thing the forecasts assume and
nobody has measured: **how research progress scales with the number and organisation of AI
researchers when experiment compute is held fixed.**

Every fast-takeoff story needs that scaling to be strong. AI 2027's superhuman coder is "30x as many
agents" at 30x speed; the December 2025 AI Futures model turns copies into progress with the rule
that the serial multiplier is about the square root of the parallel one, so doubling the copies buys
about 1.4x. The sceptical side says experiments are the bottleneck: Dwarkesh Patel to Kokotajlo,
"10 Napoleons is not 400,000 soldiers"; Noam Brown last week, research inside OpenAI goes maybe 3x
faster, held back by "running experiments serially". Both sides agree on the shape of the question
and disagree on the number.

Our experiment is that loop at toy scale. AI agents improve the training recipe of a small language
model; each experiment is a fixed 5-minute run on one GPU; the budget is 36 runs. We vary the number
of agents and how they communicate, hold compute fixed, and measure progress per run. The output is
the exponent on copies and the value of communication in an experiment-bound research loop, which is
the parameter the takeoff models plug in by assumption.

## Question

At a fixed budget of training runs, does a group of AI agents make more research progress than one
agent, and does the answer depend on how the group communicates?

Two communication structures, same budget, same task:

- **Rounds.** Lockstep: every agent proposes blind, then implements and trains, then responds to the
  measured results; the best candidate of the round becomes everyone's baseline.
- **Open.** The protocol of Park, Kontonis, Garg, Krishnamurthy and Papailiopoulos, *Scaling
  Discovery through Test-Time Communication* (arXiv 2609.21032, September 17, 2026): no rounds, no
  roles. Each agent works asynchronously in a private checkout against its share of the budget,
  declares a distinct approach in a slot, publishes findings and disconfirmations to an append-only
  shared directory, and adopts a peer's approach only after observing a clearly better measured
  result, keeping one variation of its own.

Plus the control that separates "more minds" from "more attempts": the same agents run blind to each
other (open protocol, sharing off), which is the paper's independent best@k arm at matched compute.

## Why the paper, and why now

Park et al. is the strongest evidence so far that AI collectives compound: a team of k communicating
agents matches the success rate of about 4k independent ones on ARC-AGI-3, the advantage grows with k,
and four agents beat the best human solution on MNIST classifier compression. If that transfers to AI
research, it is the mechanism a fast takeoff runs on.

But the paper states its own limit: "communication pays only when each agent has enough budget to
explore on its own", and on Terminal-Bench, with sparse feedback, two communicating agents did worse
than two independent ones. Their research tasks gave each agent 3 to 96 hours.

An ML research loop has neither of those luxuries. A training run is five minutes, the budget is
dozens of runs, and feedback is one number. That is the regime the paper warns about, and it is the
regime that matters for RSI, because it is what "AI doing AI research" looks like at any given moment:
many agents, one bottlenecked pool of experiment compute. Nobody has run the paper's protocol there,
and nobody has compared it to a lockstep protocol or to a single agent given the whole budget.

## Environment

Karpathy's autoresearch: one experiment is one 5-minute training run of a small GPT on one H100,
scored by validation bits per byte (lower is better). Agents are coding agents (Claude Code, headless)
that may change only `train.py`. Training goes through `arena-train`, which holds the single GPU lock
and claims a slot from the cell's run budget; a crash costs a slot. Every proposal, finding, run log,
commit and transcript is archived, and `analysis.verify` rejects any cell in which runs overlapped, the
budget was exceeded, or the declared winner was not the best measured run.

## Design

Every cell gets 36 training runs (about 3.3 GPU hours). Per model family:

| cell | protocol | agents | runs per agent | what it isolates |
|---|---|---|---|---|
| `*_1` | rounds, best-of-5 | 1 | 36 | one agent, the whole budget |
| `*_3`, `*_6` | rounds | 3 / 6 | 12 / 6 | lockstep collaboration vs agent count |
| `open_*_1` | open | 1 | 36 | one agent under the paper's rules |
| `open_*_3`, `open_*_6` | open | 3 / 6 | 12 / 6 | the paper's protocol vs agent count |
| `indep_*_6` | open, sharing off | 6 | 6 | six agents that cannot talk (best@6) |

Model families: Haiku 4.5 for every cell; Sonnet 5 for `_1`, `_6`, `open_6` and `indep_6`; Opus 5
only if budget allows. Seven Haiku cells are 23 GPU hours; Haiku plus the four Sonnet cells are
about 36 GPU hours, roughly $110 at RunPod prices, plus agent spend (Haiku a few dollars per cell,
Sonnet $20 to $45).

Measurements, all from the same runs:

1. **Primary:** best val_bpb after 36 runs, and the best-so-far curve against runs consumed. If time
   allows, every cell's best commit is re-run on a fresh seed so a lucky run does not decide it.
2. **Value of communication** = open_6 minus indep_6 at the same N. **Value of copies** = the
   6-agent arms minus the 1-agent arm at the same budget.
3. **Diversity:** pairwise similarity of proposals (rounds) or declared approaches (open), and how
   many approach families survive to the end. Adoption events and weak-claim labels in the open cells.
4. **Errors:** crash rate, GPU time lost to crashes, and whether a crashed idea is re-proposed by a
   different agent.
5. **Cost:** agent dollars and wall clock per cell.

## Predictions, written before the first real run

Placeholders. The team replaces them with its own numbers and commits the file before launch.

1. At 36 runs, `open_6` is within the noise-gate spread of `indep_6`: the paper's low-budget caveat
   holds in a research loop.
2. `rounds_6` ends worse than `open_6`. The lockstep discussion herds agents onto one idea (our pilot
   on Sunday night showed six agents converging on the same change after one exchange); the open
   protocol's slot and adoption rules prevent it.
3. The one-agent cells match or beat every six-agent cell at fixed compute. The group's advantage, if
   any, is in wall clock, not in progress per run.
4. The gap between `open_6` and `indep_6` is larger for Sonnet than for Haiku: communication needs
   something worth communicating.
5. Diversity: the open cells keep at least three distinct approach families to the end; the round
   cells are down to one or two families by round three.

Falsifier: if `open_6` beats both `_1` and `indep_6` by more than the noise-gate spread, then
communication pays even at research-loop budgets and the copies assumption is supported at this scale.
We say so.

## What each outcome means for takeoff

- **Communication adds nothing per run at this budget** (open ≈ independent ≈ one agent): copies buy
  wall clock, not progress per experiment. In an experiment-bound loop, RSI speed is set by compute,
  the sceptics' Amdahl reading holds, and the "30x agents" milestone is a speed claim, not a
  progress claim. Speed then has a price, payable in GPU hours, and we can state it.
- **The open protocol beats independent agents and the single agent:** communication multiplies
  progress per experiment even when each agent has only a handful of runs. That is the ingredient the
  AI 2027 5x decomposition needs (better prioritisation, less waste), measured rather than assumed,
  and the copies exponent is positive at fixed compute.
- **The premium grows with model strength** (Sonnet gap larger than Haiku gap): smarter copies
  coordinate better, which is the research-taste channel the later milestones (25x, 250x) rest on.
  If the premium shrinks with strength, copies buy the least exactly where the forecast needs them.
- **Rounds lose to open:** the organisation of the collective matters as much as its size, which is
  a design lesson for anyone building the loop and a reason forecasts should not treat "N agents" as
  one number.

Whatever comes out, it is a measured value for a parameter the two guests disagree about, at the
only scale a student group can run, with the caveats stated.

## Plan

- **Mon Sep 21.** Team on the branch; free smoke tests; one person rents an H100, runs
  `scripts/setup_gpu_box.sh --noise-gate`, and starts the two cheapest cells overnight: `haiku_1` and
  `open_haiku_1`.
- **Tue Sep 22.** The three Haiku 6-agent cells (`haiku_6`, `open_haiku_6`, `indep_haiku_6`) on three
  boxes, `haiku_3` and `open_haiku_3` on a fourth if hands allow. Sonnet cells start as soon as the
  Haiku results are above noise.
- **Wed Sep 23.** Verify every cell, fresh-seed re-runs of the winners, the figure (best-so-far
  against runs, one line per cell), slides.
- **Thu Sep 24.** Present.

Noise gate first: six runs of the unmodified baseline on the box. Any difference smaller than that
spread is not a result, whatever the plot looks like.

## What we can and cannot claim

One seed per cell, five-minute runs, one task, and Haiku is a weak researcher. Within those limits the
experiment gives the first fixed-compute comparison of a lockstep protocol, the Park et al. protocol,
independent agents and a single agent in a real training loop, and it tests the paper's own caveat in
the regime where it bites. Whatever the answer, it is a number in front of two people whose
disagreement is about exactly that number.

## Expected pushback

- **Kokotajlo:** "your agents are today's models, your budget is tiny, my copies are superhuman
  researchers." Answer: if the group premium is already negative at this scale and grows with model
  strength, that is evidence for the copies assumption; if it does not grow, it is evidence against.
  The design measures the slope, not only the level.
- **Patel:** "this says nothing about continual learning." Answer: agreed; the open protocol's shared
  directory is the closest thing to on-the-job memory in this setup, and its findings channel is what
  a memory cell would be built from.

## Where this sits in the RSI literature

Forecasts and surveys, a few measurements, none about collectives of agents in a research loop:

- **AI 2027 takeoff supplement** (Kokotajlo et al., Apr 2025): multipliers 5x, 25x, 250x, 2,000x;
  "30x as many agents" at 30x speed; 5x at fixed compute from prioritisation, smaller experiments and
  less waste.
- **AI Futures model, Dec 2025 update:** experiment compute added as an input; serial labour
  multiplier "basically the square root of parallel". Our exponent on copies is its empirical version.
- **Q2.5 2026 timelines update** (Aug 16, 2026): present coding uplift 2x median, from METR's RCT
  (1.04x to 1.2x), Anthropic's survey (4x) and Greenblatt (1.7x).
- **METR RCT** (Jul 2025): experienced developers 19% slower with AI, believing they were faster.
- **Ferreira et al.** (arXiv 2603.24647): on this same loop, a plain hyperparameter optimizer beat
  every LLM agent over long runs.
- **Dream-RSI** (Google, arXiv 2609.14858, Sep 14, 2026): weights fixed, only the controller
  improves; harness-level RSI, one agent.
- **Park et al.** (arXiv 2609.21032, Sep 17, 2026): team@k ≈ best@4k with open communication;
  pays only with enough per-agent budget and dense verification.
- **Practitioners:** Brown, about 3x, bottlenecked by serial experiments (Sep 17); Greenblatt, 4 to 5
  years of progress per year once automated (Aug 11); Sekhon, RSI "becoming a key component of the
  AI investment thesis" while "AI revenues don't sustain the capital expenditures".

This experiment is RSI at the level of the research loop with the collective as the treatment. It is
not model-level RSI: no weights are updated by the loop and agents do not learn across cells.
