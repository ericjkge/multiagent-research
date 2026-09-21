# Round {{ROUND}} — finalize and run

The debate is over. Now run your experiment.

You are in your own private checkout of the repository at `{{WORKTREE}}`, already at this round's baseline commit. Nothing you do here affects the other agents' checkouts.

## Do this

1. **Decide.** Commit to the single change you are going to make. You may keep your original proposal, adopt or adapt someone else's, or combine them — whatever you now think is most likely to lower `val_bpb`. You do not need to justify changing your mind.
2. **Read `train.py`** and implement the change. Only `train.py`.
3. **Run it** with:

       arena-train

   That is the only way to train — it holds the single GPU's lock and meters the shared compute budget. Do not invoke `train.py` directly; it will be blocked. `arena-train` waits its turn, so a pause is normal and expected. It prints `val_bpb` when it finishes.
4. **Read the result.** If it crashed, the traceback is in `run.log` (`tail -n 50 run.log`). Fix it if the bug is incidental — a typo, a shape mismatch, a missing import. Abandon it if the idea itself is broken. You get at most **{{MAX_ATTEMPTS}}** training attempts this round; each one spends shared compute, so do not brute-force.
5. **Commit** your change with `git commit -am "<short description>"`.
6. **Write `candidate.json`** in this directory:

       {"title": "...", "description": "one line, what you changed", "status": "ok" | "crash" | "abandoned", "reflection": "what you learned, for the log"}

   The orchestrator reads `val_bpb` from the run itself, not from this file — so describe honestly; there is nothing to gain by overstating.

If `arena-train` exits saying the budget is exhausted, stop: commit what you have, write `candidate.json` with status `abandoned`, and finish.

Work autonomously to the end. Do not ask for confirmation at any point.
