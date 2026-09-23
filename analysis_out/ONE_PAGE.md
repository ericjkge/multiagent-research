# Can AI agents collaborate on ML research? One page, Sep 23 2026

**Setup.** Karpathy's `autoresearch`: agents edit `train.py`, every run trains for five minutes on one H100,
score is validation bits per byte (lower is better). Every cell gets 36 runs. Two workflows: **rounds**
(blind proposals, one winner per round; Riddhi's protocol) and **open** (one long session per agent, a
shared directory of findings, scores and adoptions; Park et al. 2609.21032). Controls: agents that cannot
see each other at all. Models: Opus, Sonnet, Haiku. 23 valid cells on main, three GPU boxes.

**Read only within one box.** Untouched baselines: Monday's pods 0.9973; pod 4, Riddhi's Modal box and
Eric's box 1.012 to 1.014 (same GPU, slower hosts). Run-to-run noise 0.0008; seed-to-seed spread of a
whole cell 0.0015 to 0.0024. One seed per arm cannot detect a difference under about 0.006.

| box | cell | final | gain |
|---|---|---:|---:|
| Monday (0.9973) | Opus open 1 / 3 / 6 / 6 (seed 2) | 0.9807 / 0.9821 / 0.9779 / 0.9764 | 0.017 / 0.015 / 0.019 / 0.021 |
| Monday | Opus rounds 1 / 3 / 6 | 0.9863 / 0.9878 / 0.9805 | 0.011 / 0.010 / 0.017 |
| Monday | Sonnet open 6 | 0.9857 | 0.012 |
| pod 4 (1.0123) | Opus open 6 vs Opus isolated 6 | 0.9844 vs 0.9821 | 0.028 vs 0.030 |
| pod 4 | Sonnet isolated 6 / Haiku isolated 6 | 0.9962 / 0.9998 | 0.016 / 0.013 |
| Riddhi (1.0125) | Sonnet open 1 / 3 | 0.9793 / 0.9939 | 0.033 / 0.019 |
| Eric (1.0140) | Sonnet rounds 1 / 3 / 6 | 1.0023 / 0.9930 / 0.9780 | 0.012 / 0.021 / 0.036 |

**What the data supports**

1. **Model strength is the largest lever.** Isolated on one box: Opus 0.030, Sonnet 0.016, Haiku 0.013,
   and the weak models crash a fifth of their runs. All winners land in one basin (width over depth,
   batch 2^18, short window); the strong model finds it, the weak ones do not.
2. **Six agents beat one, three times over** (open Opus, rounds Opus, Eric's Sonnet rounds). Three agents
   do not beat one. A threshold, not a scaling law.
3. **Workflow matters as much as headcount.** Open beat rounds at 1, 3 and 6 agents. Rounds herd without
   any communication: 5 of 6 identical proposals in round 0, 6 of 6 in rounds 1 to 3, eight times the
   tokens for a worse score.
4. **Sharing was not detectable, against two different controls.** Six Opus that share vs six that see
   nothing: 0.9844 vs 0.9821 on one box (isolated slightly better, inside seed spread). Best-of-six covers
   a 36-run search as well as a team does; the paper's own caveat (communication needs enough budget per
   agent) measured in a research loop.
5. **Collaboration happened anyway.** The open logs show a collision on batch size, a called-out pivot,
   an adoption lineage across four agents, 21 published negative results. It did not buy a better number.

**For the takeoff argument.** At fixed compute the multipliers came from a stronger model and from not
wasting runs, not from copies talking to each other. This measures the organisation of automated research
at a tiny budget; it does not demonstrate recursive self-improvement.

**Caveats, stated on the slide.** Seed spread exceeds most effects: single cells prove nothing, and
"no difference" means "not detected at 0.006". Cross-box finals are not comparable. Monday's `indep_*`
cells were contaminated (agents read the shared score table and peers' commits) and are kept only as the
record of that; the isolated reruns replace them. Round cells carry a few phantom candidates at the
baseline score (harness bug, never a winner). Full detail: `case_studies.md` sections 1 to 10, the
correction at its top, `summary.md` for every cell.
