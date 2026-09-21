# Can AI agents collaborate on ML research?

CS 2881R mini-experiment, Sep 24 2026. Team: Anthony Shen, Eric Ge, Riddhi Bhagwat, Alvin Ekelund.

A multi-agent arena over [Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch).
Agents propose, argue, then implement and train — sharing one research lineage — so we can ask
whether adding agents, or diversifying them, speeds up automated ML research.

## The experiment

Each **cell** of the grid is one model family crossed with one agent count, given an identical
compute budget. Every round:

1. **Propose** — each agent reads the shared log and commits to one idea with a one-line justification.
2. **Respond** — each agent reads everyone's proposals and writes one message back to the group.
3. **Finalize** — each agent implements whatever it now believes in, trains it, and debugs its own crashes.

The best improving candidate of the round becomes the baseline **for every agent**. That shared
lineage is what makes a cell a single comparable research trajectory rather than N private
hill-climbs, and what makes the 1-agent cell a genuine control.

### Compute matching

Every cell gets the same GPU budget — 36 nominal five-minute runs, about 3.3 H100-hours — metered in
seconds so a crash that dies in 20s is not charged like a full run.

| cell | agents | BoN | candidates/round | rounds |
|---|---|---|---|---|
| `*_1` | 1 | 5 | 5 | 7 |
| `*_3` | 3 | 2 | 6 | 6 |
| `*_6` | 6 | 1 | 6 | 6 |

Best-of-N forks the agent's post-debate session N times, so the variants share context but sample
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

`--fake-gpu` simulates training, so the whole loop — parallelism, the GPU mutex, worktrees,
best-of-N, selection, archiving, resume — runs on a laptop for a few cents:

```bash
bash scripts/smoke_test.sh
```

## Reading the results

```bash
python3 -m analysis.verify  runs/<dir>   # invariants — run this FIRST
python3 -m analysis.report  runs/<dir>   # the readable write-up
python3 -m analysis.diversity runs/<dir> --classify
python3 -m analysis.errors  runs/<dir>
```

`analysis.verify` is not optional. It checks that training runs never overlapped on the GPU, that the
compute budget held, that each round's declared winner really was its best candidate, and that the
lineage never moved backwards. **A cell that fails verification is invalid, not merely weak.**

The headline plot is best `val_bpb` against *training runs consumed*, not rounds — compute is the
resource held fixed across the grid.

## How it is put together

| path | role |
|---|---|
| `orchestrator/arena.py` | the round loop: propose → respond → finalize, selection, lineage |
| `orchestrator/phases.py` | prompts and JSON schemas for the three phases |
| `orchestrator/harness.py` | headless Claude Code driver (`Harness` protocol; OpenCode slots in here) |
| `orchestrator/worktrees.py` | one git worktree per candidate, one ref per result |
| `bin/arena-train` | **the only way to train**: GPU mutex + compute meter |
| `bin/guard_hook.py` | PreToolUse hook enforcing the rules agents must not break |
| `prompts/` | the agent-facing instrument — edit deliberately, it changes the science |
| `analysis/` | verification and the four investigative analyses |

### Two design choices worth knowing

**The orchestrator owns every write to the log.** Agents read a rendered `log.md` and a
Karpathy-format `results.tsv`; their contributions arrive as schema-validated structured output. N
processes appending to one JSONL would interleave and lose records, and the log is the primary
experimental artifact. Agents keep free-form write access through `scratchpad.md`, which is
flock-guarded.

**`arena-train` is the single enforcement point.** It holds the GPU lock for the duration of a run
and meters the budget in the same critical section, so an agent cannot overspend its cell or collide
with a peer. Direct `train.py` invocation, new dependencies, edits to `prepare.py`, and anything that
leaves the box are blocked by a PreToolUse hook that fires even under `bypassPermissions`.

## Instrumentation

Captured during the run, because none of it can be reconstructed afterwards:

- every proposal and response verbatim, with per-call cost and the resolved model id;
- every `run.log`, `train.py` and `candidate.json`, archived per candidate under `runs/<dir>/rounds/`;
- a git ref per candidate commit, so any experiment can be checked out later;
- `gpu_timeline.jsonl` — start/end/queue time of every training run;
- full agent transcripts under `runs/<dir>/transcripts/` for case studies.

## Cost

Agent spend, not GPU. Roughly $5–15 per Haiku cell and $20–45 per Sonnet cell, dominated by the
finalize sessions. `max_budget_usd_per_session` caps any single agent call and `cell_budget_usd`
stops the cell; both are per-config.

## Not built yet

The Opus cells and the mixed OpenCode cell (Sonnet 5 + GPT 5.6 Terra + Gemini 3.8 Flash). Both are
additions rather than changes: heterogeneous cells are just a config listing different `model` and
`harness` per agent, and OpenCode is one more implementation of the `Harness` protocol.
