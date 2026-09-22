# Continue — you still have {{RUNS_LEFT}} training runs in your share

Your session stopped, but your share of the compute is not spent ({{CELL_LEFT}} runs left in the whole cell). Your private checkout is still `{{WORKTREE}}`, with everything you changed so far.

The shared directory now (re-read `{{LOG_PATH}}` before each decision):

<shared_directory>
{{LOG_MD}}
</shared_directory>

Pick up where you left off: next concrete change, edit `train.py`, `arena-train --title "..."`, publish what is worth knowing, adopt only on clearly better evidence and keep one variation of your own. Every run in your share must be used. When, and only when, your share is spent, write `final.json` with `"status": "finished"` and publish your final finding. Do not ask whether to continue, and never launch `arena-train` in the background.
