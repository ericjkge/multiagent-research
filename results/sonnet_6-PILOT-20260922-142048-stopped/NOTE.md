# sonnet_6 pilot, stopped 2026-09-22 ~15:30 UTC — NOT a valid cell result

Stopped manually after round 3 (18 of 36 attempts claimed, ~$58 agent spend) because of a
harness/tooling interaction that wasted roughly half of the attempts:

- With six agents queued on one GPU, `arena-train` often waits >10 min. Claude Code 2.1.278 then
  moves the command to the background and tells the agent "you will be notified when it completes".
- Sonnet 5 ends its turn to wait. In `claude -p` (non-interactive) mode that ends the session and
  kills the background job: training runs were killed mid-way (attempt consumed, no result) or
  never got the GPU (no attempt consumed).
- When a session ends without a new run, the orchestrator reads the previous round's `run.log`
  and `candidate.json` inherited from the lineage commit and records a phantom "ok" candidate at
  exactly the baseline score. (Same bug exists in the Opus cells at low rate: 1 phantom each in
  opus_6 and opus_1.) Phantoms are identifiable by a `run.log` byte-identical to an earlier round.
- The Opus agents in opus_6/opus_1 saw the same message 84 times and always kept polling
  the output file until training finished, so the Opus cells were not affected.

Fix applied before the fresh restart (see git history): agent environment raises Claude Code's
shell-tool timeout ceiling to 60 min, and the guard hook sets that timeout on `arena-train` calls
and forces them to run in the foreground. No prompt or config changes.

Box: RunPod H100 80GB HBM3, AP-IN-1, 28 vCPU quota, OMP_NUM_THREADS=8, autoresearch 228791f,
baseline val_bpb 1.014027, noise-gate spread 0.001943 (n=6).
