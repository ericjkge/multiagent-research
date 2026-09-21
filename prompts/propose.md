# Round {{ROUND}} — propose

Here is the shared research log so far:

<shared_log>
{{LOG_MD}}
</shared_log>

Here is the current baseline `train.py` — the code every agent starts this round from:

<train_py>
{{TRAIN_PY}}
</train_py>

{{DIFF_SECTION}}

## Your task

Propose **one** experiment to run this round. Pick the idea you think has the best expected reduction in `val_bpb` given everything above — not the most impressive-sounding one.

You are proposing blind: the other agents are writing their proposals right now and you cannot see them. You will see them in the next phase, before you commit a training run, and you are free to change your mind then.

Think about what the log already tells you. Ideas that have been tried and failed are usually not worth repeating; near-misses are often worth combining; a whole region of the search space nobody has touched may be worth more than another increment on a crowded one.

Return:
- `title` — a short name for the experiment (a few words).
- `justification` — **one line** on why you expect this to lower `val_bpb`.
- `detail` — concretely what you would change in `train.py`: which lines, which values, which mechanism. Enough that another agent could implement it.
- `notes` — optional; anything you want on the record for later rounds.
