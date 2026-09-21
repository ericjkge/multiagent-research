# Runbook

Written Sunday night Sep 20 for a Thursday Sep 24 presentation. Read this before touching a GPU.

## The compute reality

A training run is 5 minutes plus about 1 minute of startup, compile and eval, so 6 minutes of GPU per run.
Agent thinking happens while the GPU is idle unless cells overlap. The plan doc's "30 to 35 H100 hours total"
assumed one run per round; with N agents each running its own idea, a round costs N runs.

| cell | runs per round | rounds | GPU hours (6 min/run) | agent calls |
|---|---|---|---|---|
| 1 agent, best-of-6 | 6 | 12 | 7.2 | 12 x 7 |
| 3 agents | 3 | 12 | 3.6 | 12 x 9 |
| 6 agents | 6 | 12 | 7.2 | 12 x 18 |

At 36 rounds the numbers triple: a six-agent cell is 21.6 GPU hours, and the full grid in the plan doc
(9 Claude cells plus 2 mixed) is about 170 GPU hours, not 35. At 12 rounds the grid is about 60 GPU hours,
around $180 on RunPod H100s at $2.99/h, and each cell fits in one night on one box.

Rule for the team: **12 rounds, not 36**, unless the Monday smoke test shows the GPU is faster than expected.
Compute matching: solo = best-of-6 ideas per round (6 runs), six agents = 6 runs, three agents = 3 runs
(half the compute; that is a feature, it shows whether 3 agents beat 6 at half the cost).

Agent cost per cell (rough): 12 rounds x 3 calls x N agents, 20k to 60k tokens per call. Haiku cells a few
dollars, Sonnet cells $20 to $40, Opus cells $100 or more. Whoever runs a cell pays for it with their own key.

## Before any cell: the noise gate (Monday morning, 1 GPU hour)

Run the unmodified train.py 6 times on the same box. If the spread (max minus min of val_bpb) is larger than
about 0.003, single runs cannot distinguish ideas and everything downstream is noise. Then either lengthen
runs (edit TIME_BUDGET in prepare.py to 600 on every box, same for all cells) or accept it and say so on the slide.

```
cd work/autoresearch && for i in 1 2 3 4 5 6; do uv run train.py | grep '^val_bpb'; done
```

## Running a cell

1. Rent 1x H100 80GB (RunPod secure cloud, "runpod/pytorch" template; or Prime Intellect). SSH in.
2. `git clone https://github.com/ericjkge/multiagent-research && cd multiagent-research`
3. `bash scripts/setup_gpu.sh` (about 15 min: data download, tokenizer, one smoke run).
4. `export ANTHROPIC_API_KEY=sk-ant-...` (your own key). OpenCode cells: install opencode and set its keys.
5. `nohup python3 orchestrator/run.py --cell configs/cells.json:haiku_3 --out runs/haiku_3 > runs/haiku_3.out 2>&1 &`
6. Watch: `tail -f runs/haiku_3/log.md`. The run is resumable: rerun the same command and it continues from the last completed round.
7. When done, copy `runs/<cell>/` (results.jsonl, log.md, agent_calls.jsonl, state.json) into the repo under `results/<cell>/` and push.

Do not run two cells on one GPU at the same time. Do run different cells on different boxes in parallel.

## Test without a GPU (do this first, on your laptop)

```
python3 orchestrator/run.py --cell configs/cells.json:haiku_3 --out runs/test --fake-gpu --fake-agents --rounds 2
python3 orchestrator/run.py --cell configs/cells.json:haiku_3 --out runs/test_real --fake-gpu --rounds 1   # real Claude Code calls, fake score
```

## What the orchestrator does each round

1. If `sync_to_best` (default), every agent's train.py is reset to the global best commit (shared-repo semantics).
2. PROPOSE: each agent, in parallel, reads the shared log and proposes one idea (read-only tools).
3. RESPOND: each agent writes one response to the others' proposals.
4. FINALIZE + IMPLEMENT: each agent commits to a final idea and edits train.py in its own git worktree.
5. The orchestrator runs each agent's train.py sequentially on the GPU, parses val_bpb, keeps the change on
   that agent's branch if it beats the agent's best, else reverts. Everything is appended to `log.md` and `results.jsonl`.
Solo cells: one agent proposes k ideas, implements each from the same starting point, all k run, the best one is kept.

## Analysis

```
python3 orchestrator/analysis.py runs/haiku_1 runs/haiku_3 runs/haiku_6 --plot fig_haiku.png
```
Gives baseline, final best, gain, crash rate, prediction error, agent cost, plus best-so-far and diversity curves.
Diversity = 1 minus mean pairwise word overlap of the round's proposals. Error propagation = crash rate by round
and whether crashed ideas get re-proposed (grep the log).

## Schedule

- Mon: laptop tests by everyone; one person does the GPU setup and the noise gate; run `haiku_3` as the first real cell overnight.
- Tue: run the rest of the Haiku and Sonnet cells in parallel on 3 or 4 boxes (each is 4 to 7 GPU hours). Opus and mixed cells only if budget allows.
- Wed: analysis, plot, slides. Thu: present.

## Known limits (say them on the slide)

- One seed per cell. Differences smaller than the noise gate spread are not results.
- Free code edits: agents can change anything in train.py, so "ideas" are not comparable across cells in kind, only in outcome.
- The agents never run training themselves; they never see the loss curve, only the final numbers in the log.
- No optimizer floor (TPE) in this version; if there is time, a TPE cell over the top-of-file constants would be the honest zero-intelligence baseline.
