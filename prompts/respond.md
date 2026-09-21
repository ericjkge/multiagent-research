# Round {{ROUND}} — the results are in; respond

Everyone has run their experiment. Here is the log, including this round's proposals, every
measurement, and the round's outcome:

<shared_log>
{{LOG_MD}}
</shared_log>

Your own proposal this round was **"{{OWN_TITLE}}"**.

## Your task

Write **one** message to the group, about what the numbers actually showed.

You can dig before you write. The full results table is at `{{RESULTS_PATH}}`, and you can read any
candidate's `run.log`, `train.py` or `candidate.json` under the round directories. If you want to
know *why* someone's run crashed, or whether their gain came from the mechanism they claimed or from
something incidental, go and look. You cannot train in this phase — `arena-train` will refuse — so
the only cost of looking is your own time.

Say whatever is most useful to the group's next round. Some options, none of them obligatory:

- explain a crash someone else has not diagnosed, so the round after next does not repeat it;
- point out that a gain is inside the noise and should not be treated as a result;
- note that the winning change and your own change are orthogonal and should be combined;
- report a negative result of your own plainly, so nobody spends a slot rediscovering it;
- argue that the search is stuck in one region and name a region nobody has touched.

Be direct and specific. Vague encouragement wastes a round. Remember that only the single best
result advances, so the group's real problem is **covering the search space well**, not converging.

### Replying to a specific message

Every entry in the log has an id like `#14`. To answer one directly, or to add anything else to the
shared record:

    arena-log message --reply-to 14 "that crash was an OOM, not a shape bug — see r03/a2 run.log:88"
    arena-log message "I am dropping the rotary idea; two rounds of evidence against it"

Use it when you have something aimed at one person, or more than one thing to say. Your `response`
below is posted to the log either way.

Return:
- `response` — your message to the group.
- `notes` — optional; private notes to yourself for later rounds.
