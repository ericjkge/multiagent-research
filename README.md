# Can AI agents collaborate on ML research?

CS 2881R mini-experiment, Sep 24 2026. Team: Anthony Shen, Eric Ge, Riddhi Bhagwat, Alvin Ekelund.

Question: does adding more agents, or more diverse ones, accelerate automated ML research?
Environment: Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) (5-minute training runs of a
small LLM on one GPU, scored by val_bpb). Agents are coding agents (Claude Code headless; OpenCode stub) that
share a running log and, each round, propose, respond, finalize and edit `train.py`; the orchestrator runs the
training and records the results.

- `orchestrator/run.py` the loop (one cell per invocation, resumable)
- `orchestrator/prompts.py` the three step prompts
- `orchestrator/agents.py` CLI wrappers (claude, opencode, fake)
- `orchestrator/analysis.py` tables and plots
- `configs/cells.json` the grid (models x number of agents)
- `scripts/setup_gpu.sh` fresh-box setup
- `RUNBOOK.md` how to actually run it, and the compute arithmetic

Quick test on a laptop, no GPU, no API:

```
python3 orchestrator/run.py --cell configs/cells.json:haiku_3 --out runs/test --fake-gpu --fake-agents --rounds 2
cat runs/test/log.md
```
