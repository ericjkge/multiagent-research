# Open session — work until your share of the compute is spent

Your private checkout of the repository is `{{WORKTREE}}`, at the cell's baseline commit. You have **{{RUNS_LEFT}} training runs** in your share ({{CELL_LEFT}} left in the whole cell).

Here is the shared directory as it stands right now (it will change while you work; re-read `{{LOG_PATH}}` before each decision):

<shared_directory>
{{LOG_MD}}
</shared_directory>

## Do this, in this order

1. **Read** `train.py` and the shared directory. Note which approach families peers have already claimed. The baseline is already measured (its val_bpb and commit are at the top of the directory): do not spend a run re-measuring unchanged code; every run must test a change.
2. **Claim your slot**: `arena-log approach "<family>" "<one or two sentences: what you will explore, and what you will not assume>"`. Pick something no peer holds.
3. **Loop until your share is spent.** For each attempt:
   - decide on one concrete change, informed by the score log and the findings; do not re-run something the directory already shows failing;
   - edit `train.py` (only that file);
   - `arena-train --title "<what this attempt changes>"` and read the printed `val_bpb` (or `tail -n 50 run.log` if it crashed; fix an incidental bug, abandon a broken idea, and remember a retry costs a run);
   - if the result is worth knowing, publish it: `arena-log finding --commit <hash> "..."` (add `--weak` for a single noisy run) or `arena-log disconfirmation --commit <hash> "..."`;
   - if a peer's finding shows a clearly better result than your best, you may `arena-adopt <commit> "why"`, then keep one variation of your own;
   - if you collide with a peer, agree a convention: `arena-log coordination "..."`.
   `arena-train` tells you how many runs you have left; when it says your share is spent, stop.
4. **Finish** by writing `final.json` in `{{WORKTREE}}`:

       {"status": "finished", "best_commit": "<hash of your best run>", "best_val_bpb": <number>, "summary": "three or four sentences: what worked, what did not, what you would try next"}

   and publish one last finding that states your best result and its commit.

Work autonomously to the end. Do not stop to ask whether to continue. The only stop condition is your share being spent: every run in your share must be used, so if you run out of strong ideas, test the next most informative variation rather than stopping. Never launch `arena-train` in the background; wait for it to finish and read its result.
