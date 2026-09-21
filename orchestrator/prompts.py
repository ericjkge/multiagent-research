"""Prompts for the three agent steps. Kept in one file so the team can edit wording without touching the loop."""

RULES = """You are one of {n_agents} AI research agents improving the training of a small language model.
Environment: Karpathy's autoresearch. `train.py` in your working directory is the only file you may change.
`prepare.py` is read-only (data, tokenizer, evaluation). No new packages. The orchestrator runs training,
not you: never run train.py yourself. Each run trains for a fixed 5 minutes on one H100 and reports
val_bpb (validation bits per byte, lower is better). The goal is the lowest val_bpb.
Your agent id is `{agent_id}`. Round {round} of {n_rounds}.
"""

LOG_INTRO = """Shared log of everything all agents have proposed, discussed and measured so far
(most recent last; `keep` means that agent's branch adopted the change, `discard` means it was reverted):
"""

PROPOSE = RULES + """
{log}

Step 1 (PROPOSE). Read train.py if you need to. Propose ONE concrete change to train.py for this round.
Be specific enough that another agent could implement it. Do not repeat an idea in the log unless you say why.
Reply with ONLY a JSON object on the last line:
{{"idea": "<one or two sentences>", "justification": "<one line>"}}
"""

RESPOND = RULES + """
{log}

This round's proposals from all agents:
{proposals}

Step 2 (RESPOND). Write ONE short response to the other agents' proposals: agree, disagree, point out
a conflict or a risk, or suggest a merge. Two to four sentences. Reply with ONLY a JSON object on the last line:
{{"response": "<text>"}}
"""

FINALIZE = RULES + """
{log}

This round's proposals:
{proposals}

This round's responses:
{responses}

Step 3 (FINALIZE and IMPLEMENT). Commit to your final idea for this round (you may keep your proposal,
adopt someone else's, or merge). Then EDIT train.py in your working directory to implement it.
Keep the change minimal and make sure the file still runs. Do not run training. When you are done,
reply with ONLY a JSON object on the last line:
{{"final_idea": "<one or two sentences>", "changed_from_proposal": true/false, "predicted_val_bpb": <number>}}
"""

SOLO_PROPOSE = RULES + """
{log}

You are working alone this round. Propose {k} DIFFERENT concrete changes to train.py, each specific enough
to implement. Reply with ONLY a JSON object on the last line:
{{"ideas": [{{"idea": "...", "justification": "..."}}, ...]}}
"""

SOLO_IMPLEMENT = RULES + """
{log}

Implement exactly this idea by editing train.py in your working directory (minimal change, must still run,
do not run training): {idea}
When done reply with ONLY a JSON object on the last line:
{{"done": true, "predicted_val_bpb": <number>}}
"""
