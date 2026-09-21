# Can AI agents collaborate on ML research?

CS 2881R mini-experiment, Sep 24 2026. Team: Anthony Shen, Eric Ge, Riddhi Bhagwat, Alvin Ekelund.

A multi-agent arena over [Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch).
Agents propose, argue, then implement and train — sharing one research lineage — so we can ask
whether adding agents, or diversifying them, speeds up automated ML research.

## The experiment

Each **cell** of the grid is one model family crossed with one agent count, given an identical
compute budget. Every round:

1. **Propose** — each agent reads the shared log and commits to one idea with a one-line
   justification. It cannot see the others' proposals yet; this phase is meant to be independent.
2. **Select** — the proposals are now on the table. Each agent claims a slot from the cell's run
   budget, implements whatever it now believes in, trains it, and debugs its own crashes.
3. **Respond** — the measurements are in. Each agent reads them, can dig through any peer's
   `run.log`, and writes back to the group.

Respond comes **after** the measurement, not before it. An agent reacting to numbers is doing
something different from an agent reacting to a plan, and it is the first that this experiment is
about.

The best improving candidate of the round becomes the baseline **for every agent**. That shared
lineage is what makes a cell a single comparable research trajectory rather than N private
hill-climbs, and what makes the 1-agent cell a genuine control.

### Compute matching

Every cell gets the same budget of **training runs** — 36, about 3.3 H100-hours. A run is a run: a
crash spends a slot exactly like a success does. The cell is allotted N attempts at the GPU, and
wasting one on a bug is a real cost to the group. That is deliberate — error propagation is one of
the things under study, and it is invisible if crashes are free.

| cell | agents | BoN | slots/round | rounds if nothing crashes |
|---|---|---|---|---|
| `*_1` | 1 | 5 | 5 | 7 |
| `*_3` | 3 | 2 | 6 | 6 |
| `*_6` | 6 | 1 | 6 | 6 |

That last column is an upper bound, not a plan: a cell that crashes a lot completes fewer rounds on
the same compute. Rounds are therefore **not** comparable across cells — runs consumed are, which is
why that is the x-axis of every plot.

Best-of-N forks the agent's post-proposal session N times, so the variants share context but sample
independently.

## Running a cell

On the GPU box, once:

```bash
bash scripts/setup_gpu_box.sh     # clones autoresearch, preps data, measures the baseline once
```

It prints a commit hash — pin it as `autoresearch_commit` in every config so all cells share a
substrate. The baseline is measured once per box and reused by every cell, so no cell spends five
minutes re-measuring it and all of them start from the same number.

Then:

```bash
python3 -m orchestrator.run --config configs/haiku_3.yaml
python3 -m orchestrator.run --resume-run runs/haiku_3-20260921-140000   # after an interruption
```

Order the grid cheapest-first (`haiku_1` → `haiku_6` → `sonnet_*`); confirm the first cell's wall
clock and run count before committing 18 GPU-hours.

### Without a GPU

```bash
bash scripts/smoke_test.sh --fake-agents   # free: no API calls at all
bash scripts/smoke_test.sh                 # a few cents: real Haiku agents
```

`--fake-gpu` simulates training and `--fake-agents` simulates the researcher, so the whole loop —
parallelism, the GPU mutex, the run budget, worktrees, best-of-N, selection, archiving, resume —
runs on a laptop for nothing. The paid variant additionally exercises the prompts, structured
output, session forking and the guard hook. Run the free one after any orchestrator change; run the
paid one before touching a GPU.

## Reading the results

```bash
python3 -m analysis.verify  runs/<dir>   # invariants — run this FIRST
python3 -m analysis.report  runs/<dir>   # the readable per-cell write-up
python3 -m analysis.compare results/haiku_* --plot fig_haiku.png   # the headline figure
python3 -m analysis.diversity runs/<dir> --classify
python3 -m analysis.errors  runs/<dir>
```

`analysis.verify` is not optional. It checks that training runs never overlapped on the GPU, that
every run on the timeline claimed a slot and the run budget was never exceeded, that each round's
declared winner really was its best candidate, that the lineage never moved backwards, and that the
phases actually ran in order — no proposal logged after training began, no response logged before a
result existed. **A cell that fails verification is invalid, not merely weak.**

The headline plot is best `val_bpb` against *training runs consumed*, not rounds — compute is the
resource held fixed across the grid.

## How it is put together

| path | role |
|---|---|
| `orchestrator/arena.py` | the round loop: propose → select → respond, selection, lineage |
| `orchestrator/phases.py` | prompts and JSON schemas for the three phases |
| `orchestrator/sharedlog.py` | the collaboration medium; flock'd, numbered, agent-writable |
| `orchestrator/harness.py` | Claude Code, OpenCode and fake drivers behind one `Harness` protocol |
| `orchestrator/worktrees.py` | one git worktree per candidate, one ref per result |
| `bin/arena-train` | **the only way to train**: slot claim + GPU mutex |
| `bin/arena-log` | **the only way to write to the log**: flock'd, attributed, numbered |
| `bin/guard_hook.py` | PreToolUse hook enforcing the rules agents must not break |
| `prompts/` | the agent-facing instrument — edit deliberately, it changes the science |
| `analysis/` | verification, the per-cell report, and the cross-cell comparison |

### Two design choices worth knowing

**Agents read and write the same log.** They read a rendered `log.md` and a Karpathy-format
`results.tsv`; they write through `bin/arena-log`, which appends to the same JSONL the orchestrator
writes. Every entry is numbered, so an agent can reply to a *specific* message rather than to the
round in general. Writers are N agent processes plus the orchestrator's own threads, so every append
takes an `flock` and computes its id inside that critical section — a bare append would interleave,
and the log is the primary experimental artifact.

**`arena-train` is the single enforcement point.** It claims a run slot from the shared budget and
holds the GPU lock in the same critical section, so an agent cannot overspend its cell or collide
with a peer, and the slot is claimed *before* training so a cell can never overshoot. It also
refuses to run outside the select phase. Direct `train.py` invocation, new dependencies, edits to
`prepare.py`, and anything that leaves the box are blocked by a PreToolUse hook that fires even
under `bypassPermissions`.

## Instrumentation

Captured during the run, because none of it can be reconstructed afterwards:

- every proposal and response verbatim, with per-call cost and the resolved model id;
- every `run.log`, `train.py` and `candidate.json`, archived per candidate under `runs/<dir>/rounds/`;
- a git ref per candidate commit, so any experiment can be checked out later;
- `gpu_timeline.jsonl` — start/end/queue time of every training run;
- full agent transcripts under `runs/<dir>/transcripts/` for case studies.

## Known confounds — say these out loud

**The solo cell gets two phases, not three.** `respond` is skipped when there is nobody to respond
to, so the 1-agent control does less reflection per round as well as having fewer agents. That
conflates "more agents" with "more thinking". `solo_self_critique: true` turns respond back on for
solo cells and is the honest control; decide before running the grid, not after.

**One seed per cell.** Any difference smaller than the noise gate spread
(`scripts/setup_gpu_box.sh --noise-gate`) is not a result. Quote the spread whenever you quote a gain.

**`mixed_opencode_3` has not been run.** OpenCode has no structured-output mode, so its proposals are
parsed out of free text, and it reports no per-call spend, so `cell_budget_usd` cannot see it.

## Cost

Agent spend, not GPU. Roughly $5–15 per Haiku cell and $20–45 per Sonnet cell, dominated by the
select sessions; respond is now a tool-enabled session too, so budget a little above the old
figures. `max_budget_usd_per_session` caps any single agent call and `cell_budget_usd` stops the
cell; both are per-config.
