"""Did an "independent" agent look at its peers' work?

Scans the archived Claude Code session transcripts of a cell for tool calls that read the
shared score table, another agent's log, or another agent's commits (git show/checkout/diff/log
on a peer's commit hash). Independent cells must show none of this.

    python -m analysis.contamination results/indep_opus_6 [more cells]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .common import load_records, of_type

SESSION_RE = re.compile(r"-runs-(.+?)-work-open-(a\d+)__")


def scan(run_dir: Path) -> dict:
    records = load_records(run_dir)
    cands = of_type(records, "candidate")
    owner = {}
    for c in cands:
        if c.get("commit"):
            owner[c["commit"]] = c["agent"]
            owner[c["commit"][:7]] = c["agent"]
    winner = min((c for c in cands if c.get("status") == "ok" and c.get("val_bpb") is not None),
                 key=lambda c: c["val_bpb"], default=None)
    out = {"cell": run_dir.name, "winner": winner and (winner["agent"], winner["commit"][:7], winner["val_bpb"]),
           "agents": {}}
    # the archive can hold transcripts of another cell that ran on the same box at the same
    # time; keep only the files whose path names this cell's run directory
    try:
        tag = json.loads((run_dir / "state.json").read_text()).get("tag", "")
    except (OSError, json.JSONDecodeError):
        tag = ""
    own = tag.replace("_", "-")
    for f in sorted((run_dir / "claude_sessions").glob("*.jsonl")):
        m = SESSION_RE.search(f.name)
        if not m:
            continue
        if own and m.group(1) != own:
            continue
        me = m.group(2)
        hits = out["agents"].setdefault(me, {"results_tsv": 0, "peer_log": 0, "peer_commit": 0, "examples": []})
        for line in open(f, errors="replace"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                inp = block.get("input") or {}
                text = " ".join(str(v) for v in inp.values())
                kinds = []
                if "results.tsv" in text:
                    hits["results_tsv"] += 1; kinds.append("results.tsv")
                if re.search(r"log_a\d+\.md", text) and not re.search(rf"log_{me}\.md", text):
                    hits["peer_log"] += 1; kinds.append("peer log")
                for h in re.findall(r"\b[0-9a-f]{7,40}\b", text):
                    who = owner.get(h) or owner.get(h[:7])
                    if who and who != me:
                        hits["peer_commit"] += 1; kinds.append(f"commit {h[:7]} of {who}")
                        break
                if kinds and len(hits["examples"]) < 4:
                    hits["examples"].append((", ".join(kinds), text.replace("\n", " ")[:140]))
    return out


def main() -> int:
    for d in sys.argv[1:]:
        r = scan(Path(d))
        print(f"== {r['cell']}: winner {r['winner']}")
        for a, h in sorted(r["agents"].items()):
            flag = "LEAK" if (h["results_tsv"] or h["peer_log"] or h["peer_commit"]) else "clean"
            print(f"   {a}: {flag}  results.tsv reads={h['results_tsv']} peer-log reads={h['peer_log']} peer-commit refs={h['peer_commit']}")
            for k, ex in h["examples"]:
                print(f"       [{k}] {ex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
