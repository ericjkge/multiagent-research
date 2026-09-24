# Experiment plan and runbook

Updated September 21, 2026.

**This file is the single source of truth for the experiment matrix and its execution.**
The README describes the software; PROPOSAL.md explains the scientific question. Preserve runs
and their exact configurations.

## Question

At a fixed training-attempt allowance, how does automated ML research performance depend on
researcher model, team size, and research organization? Does sharing discoveries improve on
independent search?

## One matrix

Use the same eight conditions for each of Haiku 4.5, Sonnet 5 and Opus 5. Every condition has a
**36-attempt ceiling**, including crashes, on the same pinned autoresearch substrate. Keep medium
effort and the Claude Code harness across the matrix. No mixed-model or alternative-harness arm
in this experiment.

| Organization | Agents | Haiku config | Sonnet config | Opus config |
|---|---:|---|---|---|
| Rounds (original protocol) | 1 | `haiku_1` | `sonnet_1` | `opus_1` |
| Rounds (original protocol) | 3 | `haiku_3` | `sonnet_3` | `opus_3` |
| Rounds (original protocol) | 6 | `haiku_6` | `sonnet_6` | `opus_6` |
| Open collaboration | 1 | `open_haiku_1` | `open_sonnet_1` | `open_opus_1` |
| Open collaboration | 3 | `open_haiku_3` | `open_sonnet_3` | `open_opus_3` |
| Open collaboration | 6 | `open_haiku_6` | `open_sonnet_6` | `open_opus_6` |
| Independent control | 3 | `indep_haiku_3` | `indep_sonnet_3` | `indep_opus_3` |
| Independent control | 6 | `indep_haiku_6` | `indep_sonnet_6` | `indep_opus_6` |

**24 unique conditions: 18 rounds/open cells plus six required independent controls.**
This covers 1/3/6-agent runs under both protocols for every model and includes independent
controls at both three and six agents. Each config is one full search,
not one training attempt; the first pass is 24 searches, not nine.
Independent controls are required to make a claim about the contribution of communication;
their implementation must pass the isolation checks below before they are launched.

Independent-three is the control for open-three; independent-six is the control for open-six.
Together they distinguish the value of sharing at each team size. Independent-one is already the
open solo condition: there is no peer to communicate with. Do not duplicate that run.

### What the names mean

- **Rounds (the original protocol):** everyone proposes, gets their allotted training attempts,
  reads the results and responds; the best improvement becomes the shared starting point for the
  next round. Workers advance through the phases together.
- **Open:** each worker progresses at its own pace, reads shared findings and decides whether to
  adopt a peer's code. There is no group-wide phase barrier.
- **Independent:** the same autonomous search allocation as open, but without peer information or
  code access. The cell result is the best of the separate searches.

At three workers, open and independent allocate 12 attempts per worker; at six they allocate six.
The 36-attempt ceiling is per cell, not per worker. The rounds batching is described below.

## Comparisons and interpretation

| Priority | Comparison, within one model | Interpretation |
|---|---|---|
| Primary | `open_X_6` vs `indep_X_6` | Contribution of shared findings, scores and transferable code under the same six-agent allocation |
| Secondary communication contrast | `open_X_3` vs `indep_X_3` | Contribution of sharing under the same three-agent allocation |
| Secondary | `open_X_1` vs `open_X_3` vs `open_X_6` | Allocation of a fixed attempt allowance across one, three or six communicating researchers |
| Secondary | `open_X_1` vs `indep_X_3` vs `indep_X_6` | One long sequential search vs pools of shorter private searches |
| Secondary | `X_N` vs `open_X_N` | Performance of two implemented organizations, including their different batching, prompts and feedback schedules |
| Exploratory interaction | Compare the within-model communication differences across Haiku, Sonnet and Opus | Whether the observed communication effect differs across these researcher models |

Model tiers are not controlled parameter-count interventions. A difference between tiers does not
by itself establish a general law relating capability to collaboration.

### Preserve and disclose existing rounds behavior

The existing rounds cells use BoN=5/2/1 at N=1/3/6. The solo rounds condition is a batched search,
not the sequential open solo baseline; it also skips the response phase by default. These are
comparisons of complete research organizations, not isolated interventions on agent count or timing.

With batches of five and the current round limit, solo rounds can stop at 35 attempts despite a
36-attempt ceiling. Other cells may also finish early. Report actual counts and stop reasons, plot
progress by attempts consumed, and show comparisons at a common completed-attempt checkpoint where
needed. Never label unequal realized counts as exactly matched completed training. Preserve existing
runs; do not silently change batching or self-critique mid-grid. A redesigned rounds condition would
need a new version and separate labeling.

## Execution order: finish comparisons before adding breadth

1. Keep already-running cells and archive them with their original provenance.
2. Repair and verify independent isolation. While that work proceeds, communicating cells may run.
3. Complete the six-agent open/independent pair and open solo for each model. Prioritize
   the six-agent open/independent pair, then open solo; run each communicating cell together with
   its control so no communicating result is left without one.
4. Obtain **three fresh complete-search repetitions of each primary open-six/independent-six arm**
   for each model before expanding beyond this matrix. Replication of the primary comparison takes
   precedence over unstarted secondary cells when resources conflict. Never gate stronger-model runs
   on whether a weaker model happens to show a positive effect.
5. Complete the three-agent open/independent pairs and the remaining rounds/open cells. The
   independent-three cells are required for the three-agent communication comparison, not optional
   replacements for independent-six. Single-repetition secondary cells are descriptive; if more
   capacity remains, repeat the three-agent pairs before adding any further conditions.
6. Freeze selected recipes, independently re-evaluate them, then analyze.

Fresh repetitions use new agent sessions, empty histories, unique run IDs, and the same starting
code. Workers and successive attempts inside a cell are not independent experimental repetitions.
Do not pool them as though they were extra samples.

For each primary repetition, preferably run the communicating and independent cells sequentially
on the same GPU, with their order randomized and both orders represented across repetitions. Extra
matched GPUs allow different repetitions to run in parallel. Record GPU model/variant, baseline
measurements, software and model versions. Never run two cells simultaneously on one physical GPU:
the current GPU lock is scoped to a run directory, not to the whole machine.

### Time and budget

36 successful five-minute training attempts imply three hours of nominal training per cell, before
startup, compilation, evaluation, agent reasoning, queueing and retries. Early crashes use less GPU
time but consume attempts. Equal attempt ceilings are not equal realized GPU-seconds or equal total
AI compute; report GPU time, API usage/cost, wall time, failures and attempts separately.

- One pass through all 24 conditions: 72 nominal training GPU-hours, plus overhead (864 attempt
  slots total; actual use can be lower, as noted above).
- Three repetitions of all six primary arms (open-six and independent-six for three models), with
  one repetition of the other 18 conditions: 36 searches / 108 nominal training GPU-hours, plus overhead.
- If both the three- and six-agent open/independent arms receive three repetitions, with the other
  12 conditions run once: 48 searches / 144 nominal training GPU-hours, plus overhead.
- Pilots, baseline measurements and final recipe re-evaluation are additional and recorded separately.

Parallel hardware shortens elapsed time; agent count does not remove serialized GPU work. Measure
pilot timing and actual API spend instead of relying on an old flat cost estimate. Session ceilings
and cell ceilings are operational settings, not proof of matched inference budgets; the current
cell-dollar accounting updates after queries return and is not a strict aggregate live spend cap.

## Required launch checks and known implementation gaps

These are known issues, not claims that this documentation update fixes the harness.

- **Independent isolation is currently insufficient.** `publish_open(share=False)` filters Markdown
  but still writes a global `results.tsv`; the open prompt supplies that path to every agent. The
  whole run directory and shared Git objects also expose peer information. Independent workers need
  private score/history views and enforced separation from peer artifacts, transcripts and code.
  Test the permitted tools against a recognizable peer-only marker. No adoption events is not
  proof of no information sharing. Preserve and audit any already-run independent cells.
- **Pin the substrate and evaluator.** Configs currently leave `autoresearch_commit` blank. Record
  repository commits, CLI version, resolved model ID, effort, data/tokenizer and timing settings.
  Verify scoring and timing behavior; a writable train.py can change evaluation calls even if
  prepare.py is unchanged. Audit final diffs and use a trusted evaluator for re-evaluation.
- **Seed semantics.** YAML `seed` affects simulated behavior; it does not currently replace the real
  upstream training script's fixed torch seed. Fresh search repetitions and changed training seeds
  are different checks. Implement and verify explicit training seeds before claiming seed replication.
- **Setup repeatability.** `setup_gpu_box.sh --noise-gate` can reference an unset `VAL` when baseline.json
  already exists. Repair that path before adding the noise measurement to a previously set-up box.
  Baseline repeats measure repeatability; the hardcoded 0.003 range is not a significance test.
- **Archive repetitions uniquely.** `run_cells.sh` replaces `results/<cell>` on another run of that
  name. Use timestamped archive paths or repair the script before chaining repetitions.
- **Final selection.** Include the unchanged baseline among eligible recipes. The open summary can
  currently choose the best attempted candidate even if all candidates are worse than baseline.
- **Verification limits.** `analysis.verify` checks several accounting invariants, not isolation,
  evaluator integrity, exact quota exhaustion or the validity of the statistical conclusion.

Before a production batch, parse all intended configs, run fake smoke checks, then a short real-model,
real-GPU pilot exercising the actual worker count and multiple training opportunities. Label pilot
results and keep them out of the prespecified comparison. No runtime changes to underway runs.

## Running and preserving a cell

On a prepared GPU box, pin the agreed autoresearch commit and establish model access before launch.
The setup script installs tools and measures a baseline; resolve its known issues above before use.

```bash
bash scripts/setup_gpu_box.sh --noise-gate
mkdir -p runs results
# Example run ID; use a new value for every repetition.
python3 -m orchestrator.run --config configs/open_opus_6.yaml --run-dir runs/open_opus_6-r01
python3 -m analysis.verify runs/open_opus_6-r01
python3 -m analysis.report runs/open_opus_6-r01
```

Resume an interrupted search with `--resume-run runs/<run-id>`; a resume is not a fresh repetition.
Archive each verified run under `results/<unique-run-id>` and preserve invalid runs with their status.
Do not reuse a destination. Keep configs, provenance, model metadata, transcripts, candidate code,
logs, timing, baseline measurements and re-evaluation results. Publish results through ordinary Git
commits after reviewing the archive. A planned or claimed cell is not a completed measurement.

## Analysis

Primary outcome: quality of the recipe selected from each whole search after the attempt allowance,
including the baseline. Show search-time best validation BPB trajectories and independently
re-evaluated final quality. Choose the recipe using search-time evidence only, then retrain it and
the baseline under at least three explicit new training seeds, using the same seed set for every
condition and an untouched evaluation shard where feasible. Keep re-evaluation outside the search
allowance and never select the best re-evaluation seed.

Report every complete-search repetition and the paired open-minus-independent differences (negative
BPB difference favors communication). Three pairs are still limited descriptive evidence. Baseline
min/max spread is not a statistical decision boundary, and a noisy null does not establish equality.
Show actual resource use and any incomplete budgets next to the quality comparison.

For a mechanism case study, trace a measured finding through a message, recipient adoption, code
change and subsequent result. Inspect actual code-change families when discussing idea diversity.
Text similarity alone does not demonstrate collapse; a crash alone does not demonstrate propagated
misinformation. Existing analysis helpers may need adapting for open-protocol traces.

This studies a component of AI R&D relevant to RSI. It does not establish recursive self-improvement, a takeoff exponent, or a universal
scaling law from one small training task.
