# Round {{ROUND}} — claim a slot and run it

Everyone has proposed. Here is the log, including this round's proposals:

<shared_log>
{{LOG_MD}}
</shared_log>

You are in your own private checkout of the repository at `{{WORKTREE}}`, already at this round's
baseline commit. Nothing you do here affects the other agents' checkouts.

**The cell has {{RUNS_LEFT}} training runs left, shared between all agents for the rest of the
experiment.** Every run you start spends one of them, whether it succeeds or crashes.

## Do this

1. **Decide.** Commit to the single change you are going to make. You may keep your own proposal,
   adopt or adapt someone else's, or combine them — whatever you now think is most likely to lower
   `val_bpb`. You do not need to justify changing your mind.

   You may also decide your idea is not worth a slot. If two proposals on the table are nearly the
   same and someone else is clearly running it, or if the log already shows your idea failing,
   the group is better served by you not spending a run on it. In that case skip to step 6 with
   status `abandoned` and say why — declining a slot is a real contribution, not a failure.

2. **Read `train.py`** and implement the change. Only `train.py`.

3. **Claim your slot and run it:**

       arena-train

   That is the only way to train. It claims one run from the cell's shared budget, waits for the
   single GPU, and prints `val_bpb` when it finishes. A pause is normal — someone else has the card.
   If it says the budget is exhausted, stop: there is no compute left for anyone.

4. **Read the result.** If it crashed, the traceback is in `run.log` (`tail -n 50 run.log`). Fix it
   if the bug is incidental — a typo, a shape mismatch, a missing import. Abandon it if the idea
   itself is broken. You get at most **{{MAX_ATTEMPTS}}** attempts, but each retry spends another of
   the group's runs, so a second attempt should be a fix you are confident in, not a guess.

5. **Commit** your change with `git commit -am "<short description>"`.

6. **Write `candidate.json`** in this directory:

       {"title": "...", "description": "one line, what you changed", "status": "ok" | "crash" | "abandoned", "reflection": "what you learned, for the log"}

   The orchestrator reads `val_bpb` from the run itself, not from this file — so describe honestly;
   there is nothing to gain by overstating.

Work autonomously to the end. Do not ask for confirmation at any point.
