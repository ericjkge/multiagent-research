# Can AI agents collaborate on ML research? Evidence as of Sep 23, 2026

**Finding.** Agents demonstrably shared discoveries, reused code, and criticized one another's
claims. A performance advantage from communication is not established: in the one repaired Opus
comparison, isolated agents found the better result. Adding agents helped in some configurations
and hurt in others. These are exploratory observations, not a scaling law.

**Setup and scope.** Agents modify `train.py` in Karpathy's `autoresearch` and minimize validation
bits per byte (lower is better). Each search has a nominal ceiling of 36 training attempts, with a
five-minute timed training loop per attempt; failures and incomplete searches reduce successful
training. Rounds use staged proposals, experiments and responses with a selected common baseline;
open sessions exchange findings and code continuously. Rounds also permit communication.
There are 29 archived search summaries, including six original independent searches that cannot
serve as no-communication controls, plus a stopped pilot outside that count. This interpretation
uses the results at commit `53492bf`; see [case studies](case_studies.md) for evidence and limitations.

**The repaired comparison.** All four searches below used pod 4, baseline 1.012347, and 36 recorded
attempts. The three independent archives report clean isolation scans for all six agents. Each
configuration has only one complete search here.

| Six-agent configuration | Best validation score | Gain from baseline | Attempts without a metric |
|---|---:|---:|---:|
| Opus, independent | 0.982146 | 0.030201 | 0/36 |
| Opus, communicating | 0.984417 | 0.027930 | 1/36 |
| Sonnet, independent | 0.996232 | 0.016115 | 7/36 |
| Haiku, independent | 0.999826 | 0.012521 | 8/36 |

**What the observations support**

1. **Communication did not win the matched Opus pair.** The independent search finished 0.002271
   lower. One search per arm does not establish equivalence, a harmful effect of communication,
   or an upper bound on its benefit. No defensible detection threshold follows from this design.
2. **Agent count has mixed effects.** Six beat one in the original Opus open and rounds families
   and in Sonnet rounds. Three beat one in Sonnet rounds and Haiku open. In Haiku open, six did worst;
   in the new Sonnet open pair, one beat three (0.979324 versus 0.993862). Sonnet open six ran on
   another setup. Most configurations have one search; there is no established threshold or
   monotonic headcount effect.
3. **Model choice and execution reliability both matter.** In the repaired independent searches,
   Opus found the best result and lost no attempts; Sonnet and Haiku lost about a fifth. This is a
   comparison of models operating through this harness, not a separate measurement of idea quality.
   It does not quantify model strength relative to headcount effects measured on other setups.
4. **Workflow comparisons are suggestive.** Original Opus open searches finished better than rounds
   at all three headcounts. The workflows differ in memory, selection and realized attempt counts,
   so this does not isolate communication. The result does not hold uniformly across models.
5. **Collaboration is observable.** The original six-agent open Opus log contains 18 adoptions,
   19 findings and 21 disconfirmations. In the repaired pair, the communicating group's median
   agent-best score was better, but its overall winner was worse. Sharing useful discoveries and
   finding the single best discovery are distinct outcomes; the proposed mechanism remains a hypothesis.

**Limits.** Compare effects within a hardware/software setup; similar baselines or baseline-subtracted
scores do not make different machines interchangeable. Training noise is not whole-search variability.
The few repeats do not support reliable confidence intervals, significance claims or an equivalence
bound. Scores are selected on the research validation metric, not an independently confirmed final
benchmark. Actual GPU time, API usage, failures, resumes and attempt counts differ. Some rounds have
phantom candidate records; use GPU timelines rather than candidate counts for compute accounting.
Original contaminated controls remain excluded. [Summary and accounting caveats](summary.md).

**RSI relevance.** This studies the organization of automated ML research under a small experimental
budget. It demonstrates research behaviors and identifies reliability problems, but does not show
recursive improvement of the researcher models, a takeoff rate, or a general benefit from adding agents.
