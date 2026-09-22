# Runbook

Written Sunday night Sep 20 for a Thursday Sep 24 presentation. Read this before touching a GPU.

## The experiment grid

Every cell is 36 training runs. For each model, 1, 3 and 6 agents, under both protocols:

| model | owner | cells |
|---|---|---|
| Haiku | Eric | `haiku_1`, `haiku_3`, `haiku_6`, `open_haiku_1`, `open_haiku_3`, `open_haiku_6` |
| Sonnet | Anthony | `sonnet_1`, `sonnet_3`, `sonnet_6`, `open_sonnet_1`, `open_sonnet_3`, `open_sonnet_6` |
| Opus | Alvin | `open_opus_6` as the frontier check; `open_opus_1` if a box is free |

Optional, if a box is free: `indep_X_6` (the same six agents, unable to see each other) is the
cleanest control for whether the talking matters. Riddhi covers whatever the owners cannot fit.

Haiku cells run fine on a claude.ai login. Sonnet and Opus cells want an API key (six parallel
sessions on a login get throttled; the orchestrator waits out rate limits, but the GPU idles
meanwhile). Opus cells have $60 per session and $250 per cell ceilings in their configs.

Per box, once: `bash scripts/setup_gpu_box.sh --noise-gate` (first box) or without the flag (others),
then `export ANTHROPIC_API_KEY=...` if using a key, then
`nohup bash scripts/run_cells.sh <cell> <cell> > runs/tonight.out 2>&1 &`. In the morning:
`git pull --rebase && git push` to publish `results/`.

## The compute reality

A training run is 5 minutes plus about 1 minute of startup, compile and eval, so ~6 minutes of GPU
per run. Agent thinking happens while the GPU is idle unless cells overlap.

The grid is **compute-matched on training runs**, not on rounds. Every cell gets the same
`train_run_budget` (36 runs, ~3.3 H100-hours), and cells burn it at different rates per round:

| cell | agents | BoN | candidates/round | rounds at 36 runs | agent calls |
|---|---|---|---|---|---|
| `*_1` | 1 | 5 | 5 | 7 | 7 x 11 |
| `*_3` | 3 | 2 | 6 | 6 | 6 x 15 |
| `*_6` | 6 | 1 | 6 | 6 | 6 x 18 |

Rounds are therefore **not** a comparable x-axis between cells; training runs are. That is why
`analysis.compare` plots against runs consumed.

Full grid: 9 Claude cells x 3.3 GPU-h ≈ 30 GPU-hours, about $90 on RunPod H100s at $2.99/h, each
cell fitting in one night on one box. Agent spend is separate and larger for the big models: a few
dollars per Haiku cell, $20–45 per Sonnet cell, $100+ per Opus cell. Whoever runs a cell pays for it
with their own key.

## Before any cell: the noise gate

**Do this first. Nothing downstream means anything without it.**

```bash
bash scripts/setup_gpu_box.sh --noise-gate
```

It runs the unmodified `train.py` six times on the same box and writes `noise_gate.json`. If the
spread (max − min of `val_bpb`) is larger than about 0.003, single runs cannot distinguish ideas and
every number in the grid is noise. Then either lengthen runs (`TIME_BUDGET` in `prepare.py`, the
**same value on every box and every cell**) or accept it and say so explicitly on the slide.

Quote the measured spread whenever you quote a gain.

## Running a cell

1. Rent 1x H100 80GB (RunPod secure cloud, "runpod/pytorch" template; or Prime Intellect). SSH in.
2. `git clone https://github.com/ericjkge/multiagent-research && cd multiagent-research`
3. `export ANTHROPIC_API_KEY=sk-ant-...` — an API key, not a claude.ai login. The cells run
   unattended; an expired OAuth token fails every call in the cell.
4. `bash scripts/setup_gpu_box.sh --noise-gate` (~45 min: toolchain, data, baseline, noise gate).
   It prints a commit hash. **Pin it as `autoresearch_commit` in every config** so all cells share a
   substrate.
5. Launch:

   ```bash
   nohup python3 -m orchestrator.run --config configs/haiku_3.yaml > runs/haiku_3.out 2>&1 &
   ```

6. Watch: `tail -f runs/haiku_3-*/log.md`.
7. Interrupted? `python3 -m orchestrator.run --resume-run runs/haiku_3-20260921-140000`.
8. When it finishes, **verify before believing anything**, then archive:

   ```bash
   python3 -m analysis.verify runs/haiku_3-<stamp>     # must print OK
   python3 -m analysis.report runs/haiku_3-<stamp>
   cp -r runs/haiku_3-<stamp> results/haiku_3 && git add results/haiku_3 && git commit
   ```

`runs/` is gitignored (it holds worktrees and transcripts); `results/` is committed. Copy the cell
directory across when it is done and verified.

Do not run two cells on one GPU at the same time — `arena-train`'s lock is per run directory, not
per machine, so two concurrent cells would overlap on the card and both sets of timings would be
void. Different cells on different boxes in parallel is exactly right.

Order the grid cheapest-first (`haiku_1` → `haiku_6` → `sonnet_*` → `opus_*`) and confirm the first
cell's wall clock and run count before committing the rest.

## Testing without a GPU

Two levels, both on a laptop:

```bash
# free: fake GPU AND fake agents. Exercises worktrees, the GPU mutex, the run
# budget, selection, archiving and resume. No API calls, no money.
bash scripts/smoke_test.sh --fake-agents

# a few cents: real Claude Code agents, simulated training. Also exercises
# prompts, structured output, session forking and the guard hook.
bash scripts/smoke_test.sh
```

Run the free one after any orchestrator change. Run the paid one before touching a GPU.

## What the orchestrator does each round

1. Every agent starts the round from the same baseline commit — the shared lineage.
2. **Propose** — each agent, in parallel, reads the shared log and commits to one idea.
3. **Select** — each agent claims a slot from the run budget, implements its idea in its own git
   worktree, and trains it via `arena-train`, which serializes onto the one GPU.
4. **Respond** — each agent reads the round's results and writes back to the group.
5. The best *improving* candidate becomes the new baseline for **everyone**.

`bin/arena-train` is the single enforcement point: it holds the GPU lock, claims a slot from the run
budget in the same critical section, and refuses once the budget is gone. `bin/guard_hook.py` is a
PreToolUse hook that blocks direct `train.py` invocation, edits to `prepare.py`, new dependencies and
anything leaving the box — it fires even under `bypassPermissions`.

## The open protocol (Park et al. 2609.21032)

`configs/open_*.yaml` and `configs/indep_*.yaml` run the second protocol on the same 36-run budget.
Differences that matter when running them:

- Each agent is one long Claude Code session, resumed while its share of runs lasts
  (`open_max_resumes`, `open_session_timeout_s`). Agent cost per cell is similar to the round
  protocol; raise `max_budget_usd_per_session` for Sonnet.
- Nothing is orchestrated between runs: the agent decides when to train, what to publish and whether
  to adopt. Watch `runs/<dir>/log.md` change live.
- The cell's result is the best measured run, whoever made it. `analysis.verify` additionally checks
  that every GPU run is on the score log, that no agent exceeded its share, and that an independent
  cell contains no adoption events.
- The independent control (`indep_*`) gives each agent its own `log_<agent>.md`; agents cannot see
  each other at all. Compare it against `open_*` with the same N to get the value of communication,
  and both against `*_1` to get the value of copies.

Smoke tests: `bash scripts/smoke_test.sh --config configs/smoke_open.yaml --fake-agents` (free) and
without `--fake-agents` (a few dollars of Haiku), plus `configs/smoke_open_indep.yaml`.

## Analysis

```bash
python3 -m analysis.verify  runs/<dir>          # invariants — run this FIRST
python3 -m analysis.report  runs/<dir>          # the readable per-cell write-up
python3 -m analysis.compare results/haiku_* --plot fig_haiku.png   # the headline figure
python3 -m analysis.diversity runs/<dir> --classify
python3 -m analysis.errors  runs/<dir>
```

`analysis.verify` is not optional. **A cell that fails it is invalid, not merely weak.**

Diversity = mean pairwise TF-IDF cosine between a round's proposals; rising over the cell is the
signature of collapse. Error propagation = crash rate by round, plus whether a crashed idea gets
re-proposed by *other* agents next round.

## Known limits — say these on the slide

- One seed per cell. Differences smaller than the noise gate spread are not results.
- Agents can change anything in `train.py`, so ideas are not comparable across cells in kind, only
  in outcome.
- Agents never see the loss curve, only the final numbers in the log.
- No optimizer floor. A TPE sweep over the top-of-file constants would be the honest
  zero-intelligence baseline; if there is time, run one.
- `mixed_opencode_3` has not been run. OpenCode has no structured-output mode, so its proposals are
  parsed out of free text and its spend is not metered.

## Schedule

- **Mon**: laptop tests by everyone (`--fake-agents` is free). One person does the GPU setup and the
  noise gate. Start `haiku_3` overnight as the first real cell.
- **Tue**: the rest of the Haiku and Sonnet cells in parallel on 3–4 boxes. Opus only if budget allows.
- **Wed**: analysis, figures, slides.
- **Thu**: present.
