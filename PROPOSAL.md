# Can AI agents collaborate on ML research?

CS 2881R mini-experiment, September 24, 2026.
Team: Anthony Shen, Eric Ge, Riddhi Bhagwat, Alvin Ekelund.

## Research question

When training experiments are costly, does sharing discoveries help AI researchers make better use
of a fixed training-attempt allowance? How do the results vary with researcher model, team size and
research organization?

AI research automation is one component of recursive self-improvement scenarios. Copies of a
researcher could explore complementary ideas, reuse discoveries and avoid repeated failures. They
could also duplicate work, adopt erroneous conclusions or spend scarce time coordinating. We study
these mechanisms in a small language-model training loop. The research agents and their controller
remain fixed: improving a target model's recipe does not demonstrate recursive improvement of the
researcher itself.

## Design

**The [RUNBOOK.md matrix](RUNBOOK.md#one-matrix) is the authoritative plan.** It specifies identical
conditions for Haiku 4.5, Sonnet 5 and Opus 5: rounds and open protocols at 1/3/6 agents, plus required
independent-three and independent-six controls for each model. There are 24 distinct conditions. Existing teammate runs are
preserved; the same design extends to Opus rather than treating it as an unmatched showcase.

The primary contrast is open-six versus independent-six within each model. A planned secondary
contrast compares open-three with independent-three, testing sharing at the smaller team size.
Open solo also serves as independent solo, since it has no peers. The independent control
must prevent access to peer findings, scores and code; the runbook records a current isolation defect
that must be repaired before this contrast is valid. Open solo is the sequential baseline. Rounds
cells are secondary comparisons of complete organizations: current batching and reflection behavior
also differ, so their contrast with open cells does not isolate communication timing.

Each condition has a 36-attempt ceiling on a pinned autoresearch substrate; failures consume attempts.
Actual completed attempts, GPU time, API usage and wall time are reported separately. The current
rounds solo configuration can finish at 35 attempts, so equal ceilings must not be mislabeled as equal
completed training. See the runbook for ownership, execution priority, replication and launch checks.

## Related work and intended contribution

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) provides the small, fixed-duration
  language-model training loop and validation-bits-per-byte objective.
- [Shen et al., An Empirical Study of Multi-Agent Collaboration for Automated Research](https://arxiv.org/abs/2603.29632)
  already compares multi-agent autoresearch topologies under computational constraints. We do not
  claim the first fixed-budget study of AI research teams.
- [Park et al., Scaling Discovery through Test-Time Communication](https://arxiv.org/abs/2609.21032)
  studies verified progress sharing, independent controls and a coordination penalty at low budgets.
  Our study examines a setting with few, expensive and potentially noisy training measurements.
- [Dream-RSI](https://arxiv.org/abs/2609.14858) improves and redeploys an exploration policy while keeping
  the underlying coding agent fixed. Our experiment holds the research policy fixed and compares
  organizations; it does not implement that recursive policy-improvement loop.

The intended contribution is a controlled, replicated class-scale measurement of communication in
this training setting, accompanied by evidence from the actual research traces. It is not a novelty
claim about multi-agent research generally.

## Hypotheses and measurement

Competing mechanisms make the outcome uncertain:

1. Shared measured discoveries could improve final recipe quality relative to six independent searches.
2. Six attempts per worker could be too few to recover the coordination cost or compound discoveries.
3. Shared code and findings could reduce wasted attempts, but also reduce exploration diversity.
4. The communication effect may vary across models; neither its sign nor a monotonic trend is assumed.

Record the protocol and any numerical predictions before examining the corresponding outcomes.
Runs already inspected cannot retroactively become a preregistered experiment.

Target three fresh complete-search repetitions per primary arm and model. The experimental unit is
the whole search, not each agent or each training attempt. Select final recipes using search-time
validation, including the unchanged baseline, and re-evaluate the frozen recipes and baseline using
new explicit training seeds and an untouched evaluation shard where feasible. Report all search
repetitions and paired differences; three pairs still support only limited conclusions.

Secondary measurements: best-so-far curves by attempts consumed, resource use, failure rates, and
trace-based case studies of adoption or error transfer. Code-change inspection complements textual
similarity when assessing diversity. The baseline repeatability range is a diagnostic, not a
significance threshold.

## Scope of conclusions

A positive result would support useful discovery sharing in this model/task/budget regime. A null or
negative result would constrain the benefit observed here, without ruling out larger budgets or
other organizations. Differences across models would be evidence about these models, not a causal
law of model size. Five-minute training recipes need not transfer to longer training.

We do not infer a takeoff rate, a copies-scaling exponent or superhuman-researcher behavior from this
experiment. The presentation will separate measured outcomes, observed mechanisms and speculation.
