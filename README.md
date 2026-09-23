# Can AI agents collaborate on ML research?

CS 2881R mini-experiment, September 24, 2026.
Team: Anthony Shen, Eric Ge, Riddhi Bhagwat, Alvin Ekelund.

An arena over [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) for comparing
researcher models, team sizes and research organizations under a fixed training-attempt allowance.

**[RUNBOOK.md](RUNBOOK.md) is the single source of truth for the matrix, ownership, priorities,
launch checks and analysis.** [PROPOSAL.md](PROPOSAL.md) gives the scientific framing and prior work.
The matrix has 24 conditions: Haiku/Sonnet/Opus, each with rounds and open 1/3/6 plus independent
3/6. Open solo also serves as independent solo; there is no duplicate one-agent control.
Independent controls are required for the communication claim, but their isolation needs repair
before production use. See the runbook's known implementation gaps; a config existing is not proof
that a scientifically valid run has completed.

## Protocols

- **Rounds (the original protocol):** agents propose, implement and train, then respond to results.
  Everyone advances through these phases together. The best improving
  candidate becomes the shared baseline. Existing BoN is 5/2/1 for 1/3/6 agents; solo response is
  disabled by default. Interpret this as a complete research organization, including its batching.
- **Open:** agents operate in long autonomous sessions, each with a share of the attempt allowance.
  They publish approaches, findings and scores and can adopt peer code. The protocol is inspired by
  [Park et al.](https://arxiv.org/abs/2609.21032).
- **Independent:** intended open-protocol control with private histories and no peer access.
  `open_share_log: false` currently filters Markdown but still exposes the shared score table and
  other artifacts; it is not yet sufficient isolation.

`arena-train --title "..."` claims an attempt and serializes training through a per-cell GPU lock.
`arena-log` records messages; `arena-adopt` records an explicit transfer of peer code. Read the
runbook before launching, particularly the prohibition on simultaneous cells on the same GPU.

## Entry points

```bash
# Free simulated smoke checks (the script creates a fake substrate).
bash scripts/smoke_test.sh --fake-agents
bash scripts/smoke_test.sh --config configs/smoke_open.yaml --fake-agents
bash scripts/smoke_test.sh --config configs/smoke_open_indep.yaml --fake-agents

# After the runbook's production checks and substrate pinning:
python3 -m orchestrator.run --config configs/open_opus_6.yaml --run-dir runs/open_opus_6-r01
python3 -m orchestrator.run --resume-run runs/open_opus_6-r01

# Inspect a completed run before treating it as evidence.
python3 -m analysis.verify runs/open_opus_6-r01
python3 -m analysis.report runs/open_opus_6-r01
```

Fake tests exercise orchestration, not real model access, training quality or information isolation.
The verifier checks accounting invariants; separate checks are needed for peer isolation, evaluator
integrity and actual completed budgets. Preserve each repetition in a unique archive directory.

## Implementation map

| Path | Purpose |
|---|---|
| `orchestrator/arena.py` | Rounds, candidate selection and shared lineage |
| `orchestrator/open_arena.py` | Autonomous sessions for open/independent configurations |
| `orchestrator/config.py` | Config schema, quotas and attempt ceilings |
| `orchestrator/harness.py` | Claude Code, OpenCode and fake adapters |
| `orchestrator/sharedlog.py` | Recorded events and rendered agent views |
| `orchestrator/worktrees.py` | Candidate worktrees and Git references |
| `bin/arena-train` | Attempt accounting and GPU serialization |
| `bin/arena-log`, `bin/arena-adopt` | Messages and recorded code adoption |
| `bin/guard_hook.py` | Tool-use checks; not an OS isolation boundary |
| `prompts/` | Agent instructions; version changes as experimental changes |
| `analysis/` | Verification, reports, trajectories and exploratory analysis |

Runs record proposals/messages, candidate code and logs, Git commits, agent transcripts, timing,
configuration and model metadata. `runs/` is ignored by Git; archive reviewed results under unique
`results/<run-id>` paths. API usage and actual GPU time must accompany training-attempt counts.
