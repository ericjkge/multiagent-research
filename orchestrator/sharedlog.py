"""The shared log: proposals, messages, experiments and results.

This is the collaboration medium.  Agents have **read and write access**: they
read a rendered Markdown view plus a Karpathy-format ``results.tsv``, and they
write through ``bin/arena-log``, which appends to the same JSONL the
orchestrator writes.

Because writers are now N agent processes *plus* the orchestrator's own
threads, every append takes an ``flock`` on the JSONL and computes the record's
id inside that critical section.  A bare ``a``-mode write is not enough: two
processes would interleave and the log is the primary experimental artifact.

Every record carries a monotonic integer ``id``, rendered next to the entry in
``log.md``, which is what lets an agent reply to a *specific* message rather
than to the round in general.

Record types: ``round_start``, ``proposal``, ``message``, ``response``,
``note``, ``candidate``, ``round_end``.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable

MAX_DETAIL_CHARS = 1200


def append_record(jsonl: Path, kind: str, **fields: Any) -> dict:
    """Append one record, cross-process safe, assigning it the next id.

    Shared by the orchestrator and by ``bin/arena-log`` so that agent writes
    and orchestrator writes cannot corrupt each other.
    """
    jsonl = Path(jsonl)
    jsonl.touch(exist_ok=True)
    with open(jsonl, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            # The id is the count of existing records; computing it under the
            # same lock as the write is what makes it unique.
            existing = sum(1 for line in fh if line.strip())
            record = {"id": existing + 1, "t": kind, "ts": time.time(), **fields}
            fh.seek(0, os.SEEK_END)
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return record


class SharedLog:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.jsonl = self.run_dir / "log.jsonl"
        self.markdown = self.run_dir / "log.md"
        self.results_tsv = self.run_dir / "results.tsv"
        self.scratchpad = self.run_dir / "scratchpad.md"
        self._lock = threading.Lock()
        self.jsonl.touch(exist_ok=True)

    # -- writing ---------------------------------------------------------

    def append(self, kind: str, **fields: Any) -> dict:
        # The threading lock keeps the orchestrator's own threads off each
        # other; append_record's flock keeps agent processes off all of them.
        with self._lock:
            return append_record(self.jsonl, kind, **fields)

    def records(self) -> list[dict]:
        if not self.jsonl.exists():
            return []
        out = []
        for line in self.jsonl.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def append_scratchpad(self, agent: str, text: str) -> None:
        """Free-form agent writes, serialized across processes."""
        self.scratchpad.touch(exist_ok=True)
        with open(self.scratchpad, "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(f"\n### {agent} @ {time.strftime('%H:%M:%S')}\n{text.strip()}\n")
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    # -- rendering the agent-facing view ---------------------------------

    def publish(
        self,
        upto_round: int,
        include_proposals_for: int | None = None,
        include_results_for: int | None = None,
        budget: dict | None = None,
    ) -> None:
        """Regenerate the files agents read.  Called before each phase."""
        records = self.records()
        self.markdown.write_text(
            render_log_md(
                records, upto_round, include_proposals_for, include_results_for, budget
            )
        )
        self.results_tsv.write_text(render_results_tsv(records))

    def publish_open(self, budget: dict | None = None, agents: list[str] | None = None,
                     share: bool = True) -> None:
        """Regenerate the agent-facing view under the open protocol.

        Called by the orchestrator at start and by ``arena-train`` /
        ``arena-log`` / ``arena-adopt`` after every write, because under the
        open protocol there is no phase boundary at which the orchestrator
        could do it.  With ``share`` off each agent gets its own ``log_<id>.md``
        holding only its own entries: the independent-agents control.
        """
        records = self.records()
        if budget is None:
            bp = self.run_dir / "budget.json"
            budget = json.loads(bp.read_text()) if bp.exists() else None
        if share:
            self.results_tsv.write_text(render_results_tsv(records))
            self.markdown.write_text(render_open_log_md(records, budget))
            return
        # independent: no global table while the cell runs (agents were reading it), one
        # score table per agent holding only its own runs
        if self.results_tsv.exists():
            self.results_tsv.unlink()
        for agent in agents or sorted({r.get("agent") for r in records if r.get("agent")}):
            mine = [r for r in records if r.get("agent") in (agent, None, "")]
            (self.run_dir / f"results_{agent}.tsv").write_text(render_results_tsv(mine))
            (self.run_dir / f"log_{agent}.md").write_text(
                render_open_log_md(mine, budget, only_agent=agent)
            )


def open_log_path(run_dir: Path, agent: str, share: bool) -> Path:
    return Path(run_dir) / ("log.md" if share else f"log_{agent}.md")


def render_open_log_md(records: list[dict], budget: dict | None = None,
                       only_agent: str | None = None) -> str:
    """The shared directory of the open protocol, rendered as one document.

    Mirrors the layout of Park et al.: approaches (the slots), findings (an
    append-only broadcast channel), disconfirmations, a score log with one
    line per attempt, coordination notes, and adoption events.
    """
    by = {}
    for r in records:
        by.setdefault(r.get("t"), []).append(r)

    out: list[str] = ["# Shared research directory (open protocol)", ""]
    if only_agent:
        out += [f"_You are `{only_agent}`. This cell runs agents independently: you see only "
                "your own entries and results. There are no peers to read or reach._", ""]
    starts = by.get("round_start", [])
    if starts:
        out += [f"Baseline at start of the cell: **val_bpb {starts[0].get('baseline_bpb', 0):.6f}** "
                f"(commit {str(starts[0].get('baseline_commit', ''))[:7]})", ""]
    if budget:
        left = budget.get("budget_runs", 0) - budget.get("runs", 0)
        quota = budget.get("per_agent_quota") or 0
        out.append(f"**Training runs remaining in this cell: {left} of {budget.get('budget_runs', 0)}.**")
        if quota:
            used = {}
            for c in budget.get("claims", []):
                used[c["agent"]] = used.get(c["agent"], 0) + 1
            shares = ", ".join(f"{a}: {quota - n} left" for a, n in sorted(used.items()))
            out.append(f"Each agent's share is {quota} runs. Used so far: {shares or 'none yet'}.")
        out.append("")

    out += ["## Approaches (slots)", ""]
    approaches = [p for p in by.get("proposal", []) if p.get("subtype") == "approach"]
    if approaches:
        for p in sorted(approaches, key=lambda x: x.get("id", 0)):
            out.append(f"- `#{p.get('id', '?')}` **{p['agent']}** — {p.get('title', '')}: "
                       f"{_truncate(p.get('detail', ''), 600)}")
    else:
        out.append("_No approach claimed yet._")
    out.append("")

    out += ["## Score log (one line per attempt, in GPU order)", "",
            "| # | agent | val_bpb | status | commit | attempt |", "|---|---|---|---|---|---|"]
    best = None
    for c in sorted(by.get("candidate", []), key=lambda x: x.get("round", 0)):
        vb = c.get("val_bpb")
        metric = f"{vb:.6f}" if (c.get("status") == "ok" and vb is not None) else "—"
        if c.get("status") == "ok" and vb is not None and (best is None or vb < best[0]):
            best = (vb, c.get("agent"), c.get("commit", ""))
        out.append(f"| {c.get('round', '?')} | {c.get('agent', '?')} | {metric} | {c.get('status', '?')} | "
                   f"`{str(c.get('commit', ''))[:7]}` | {_truncate(c.get('description') or c.get('title', ''), 160)} |")
    if best:
        out += ["", f"**Best so far: val_bpb {best[0]:.6f} by {best[1]} at commit `{best[2][:7]}`.**"]
    out.append("")

    def section(title: str, kind: str, empty: str) -> None:
        out.append(f"## {title}")
        out.append("")
        items = by.get(kind, [])
        if not items:
            out.append(f"_{empty}_")
        for m in sorted(items, key=lambda x: x.get("id", 0)):
            tag = " _(weak claim)_" if m.get("weak") else ""
            ref = f" [commit `{str(m.get('commit'))[:7]}`]" if m.get("commit") else ""
            reply = f" _(re: #{m.get('reply_to')})_" if m.get("reply_to") else ""
            out.append(f"- `#{m.get('id', '?')}` **{m.get('agent', '?')}**{reply}{tag}{ref}: "
                       f"{_truncate(m.get('text', ''), 900)}")
        out.append("")

    section("Findings (append-only broadcast)", "finding", "No findings published yet.")
    section("Disconfirmations (negative results, attempts to falsify)", "disconfirmation",
            "No disconfirmations yet.")
    adoptions = by.get("adoption", [])
    out += ["## Adoption events", ""]
    if adoptions:
        for a in sorted(adoptions, key=lambda x: x.get("id", 0)):
            out.append(f"- `#{a.get('id', '?')}` **{a['agent']}** adopted `{str(a.get('commit', ''))[:7]}` "
                       f"from {a.get('from_agent', '?')}: {_truncate(a.get('text', ''), 300)}")
    else:
        out.append("_No adoptions yet._")
    out.append("")
    section("Coordination (conventions agreed after collisions)", "coordination", "Nothing yet.")
    section("Messages", "message", "No messages.")
    return "\n".join(out) + "\n"


def _truncate(text: str, limit: int = MAX_DETAIL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def _render_messages(msgs: list[dict]) -> list[str]:
    """Messages in the order they were written, with reply targets shown.

    Threading is rendered as an explicit "re: #n" rather than by nesting: an
    agent reads this as flat text in a prompt, and indentation is a weaker
    signal there than the id itself.
    """
    out: list[str] = []
    for m in sorted(msgs, key=lambda x: x.get("id", 0)):
        text = _truncate(m.get("text", ""), 900)
        reply = m.get("reply_to")
        tag = f" _(re: #{reply})_" if reply else ""
        out.append(f"- `#{m.get('id', '?')}` **{m.get('agent', '?')}**{tag}: {text}")
    return out


def render_results_tsv(records: Iterable[dict]) -> str:
    """Karpathy's five-column format, so the substrate stays familiar."""
    lines = ["commit\tval_bpb\tmemory_gb\tstatus\tdescription"]
    for r in records:
        if r["t"] != "candidate":
            continue
        vram_gb = (r.get("peak_vram_mb") or 0.0) / 1024
        lines.append(
            "\t".join(
                [
                    (r.get("commit") or "-------")[:7],
                    f"{r.get('val_bpb') or 0.0:.6f}",
                    f"{vram_gb:.1f}",
                    r.get("status", "crash"),
                    (r.get("description") or r.get("title") or "").replace("\t", " "),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def render_log_md(
    records: list[dict],
    upto_round: int,
    include_proposals_for: int | None = None,
    include_results_for: int | None = None,
    budget: dict | None = None,
) -> str:
    """The running log as the agents see it.

    Rounds strictly before ``upto_round`` are rendered in full.  The current
    round is revealed in stages: its proposals once everyone has proposed, its
    results once everyone has run.  That staging is what makes propose
    independent and respond informed -- an agent proposing must not see its
    peers' proposals, and an agent responding must see the measurements.
    """
    by_round: dict[int, dict[str, list[dict]]] = {}
    for r in records:
        rnd = r.get("round")
        if rnd is None:
            continue
        slot = by_round.setdefault(rnd, {})
        slot.setdefault(r["t"], []).append(r)

    out: list[str] = ["# Shared research log", ""]

    starts = [r for r in records if r["t"] == "round_start"]
    if starts:
        first = starts[0]
        out += [
            f"Baseline at start of the cell: **val_bpb {first.get('baseline_bpb', 0):.6f}**",
            "",
        ]

    if budget:
        left = budget.get("budget_runs", 0) - budget.get("runs", 0)
        out += [
            f"**Training runs remaining in this cell: {left} of "
            f"{budget.get('budget_runs', 0)}.** This is the shared, fixed budget for "
            "the whole cell; every run any agent starts spends one, crash or not.",
            "",
        ]

    for rnd in sorted(k for k in by_round if k < upto_round):
        slot = by_round[rnd]
        start = (slot.get("round_start") or [{}])[0]
        out.append(f"## Round {rnd}")
        out.append(
            f"_Baseline entering the round: val_bpb {start.get('baseline_bpb', 0):.6f} "
            f"(commit {str(start.get('baseline_commit', ''))[:7]})_"
        )
        out.append("")

        if slot.get("proposal"):
            out.append("### Proposals")
            for p in sorted(slot["proposal"], key=lambda x: x["agent"]):
                out.append(f"- `#{p.get('id', '?')}` **{p['agent']} — {p.get('title', '')}**")
                out.append(f"  - why: {_truncate(p.get('justification', ''), 300)}")
                detail = _truncate(p.get("detail", ""))
                if detail:
                    out.append(f"  - detail: {detail}")
            out.append("")

        if slot.get("response") or slot.get("message"):
            out.append("### Messages")
            out += _render_messages(
                (slot.get("response") or []) + (slot.get("message") or [])
            )
            out.append("")

        if slot.get("candidate"):
            out.append("### Experiments run")
            for c in sorted(slot["candidate"], key=lambda x: (x["agent"], x.get("variant", 0))):
                label = f"{c['agent']}/v{c.get('variant', 0)}"
                # Anything without a metric failed, whether it crashed outright
                # or the agent abandoned the idea.
                if c.get("status") != "ok" or c.get("val_bpb") is None:
                    out.append(
                        f"- `{label}` **{c.get('status', 'crash').upper()}** — "
                        f"{_truncate(c.get('description', ''), 200)}"
                        f" ({_truncate(c.get('error', ''), 160)})"
                    )
                else:
                    vram = c.get("peak_vram_mb") or 0.0
                    out.append(
                        f"- `{label}` val_bpb **{c['val_bpb']:.6f}** "
                        f"({vram / 1024:.1f} GB) — "
                        f"{_truncate(c.get('description', ''), 200)}"
                    )
            out.append("")

        end = (slot.get("round_end") or [{}])[0]
        if end:
            if end.get("improved"):
                w = end.get("winner", {})
                out.append(
                    f"**Outcome:** {w.get('agent')}/v{w.get('variant')} won; new baseline "
                    f"val_bpb {end.get('new_baseline_bpb', 0):.6f}."
                )
            else:
                out.append(
                    "**Outcome:** nothing beat the baseline; it is unchanged at "
                    f"val_bpb {end.get('new_baseline_bpb', 0):.6f}."
                )
            out.append("")

    if include_results_for is not None:
        slot = by_round.get(include_results_for, {})
        cands = slot.get("candidate", [])
        if cands:
            out.append(f"## Round {include_results_for} — results, just in")
            out.append("")
            for c in sorted(cands, key=lambda x: (x["agent"], x.get("variant", 0))):
                label = f"{c['agent']}/v{c.get('variant', 0)}"
                if c.get("status") != "ok" or c.get("val_bpb") is None:
                    out.append(
                        f"- `{label}` **{c.get('status', 'crash').upper()}** — "
                        f"{_truncate(c.get('description', ''), 200)}"
                        f" ({_truncate(c.get('error', ''), 300)})"
                    )
                else:
                    vram = c.get("peak_vram_mb") or 0.0
                    out.append(
                        f"- `{label}` val_bpb **{c['val_bpb']:.6f}** ({vram / 1024:.1f} GB) — "
                        f"{_truncate(c.get('description', ''), 200)}"
                    )
                    if c.get("reflection"):
                        out.append(f"  - {c['agent']} says: {_truncate(c['reflection'], 400)}")
            out.append("")

            end = (slot.get("round_end") or [{}])[0]
            if end.get("improved"):
                w = end.get("winner", {})
                out.append(
                    f"**Outcome:** {w.get('agent')}/v{w.get('variant')} wins the round; the "
                    f"new baseline for everyone is val_bpb {end.get('new_baseline_bpb', 0):.6f}."
                )
            elif end:
                out.append(
                    "**Outcome:** nothing beat the baseline; it is unchanged at "
                    f"val_bpb {end.get('new_baseline_bpb', 0):.6f}."
                )
            out.append("")

        msgs = slot.get("message", [])
        if msgs:
            out.append("### Messages already sent this round")
            out += _render_messages(msgs)
            out.append("")

    if include_proposals_for is not None:
        current = by_round.get(include_proposals_for, {}).get("proposal", [])
        if current:
            out.append(f"## Round {include_proposals_for} — proposals on the table now")
            out.append("")
            for p in sorted(current, key=lambda x: x["agent"]):
                out.append(f"### `#{p.get('id', '?')}` {p['agent']}: {p.get('title', '')}")
                out.append(f"*Justification:* {_truncate(p.get('justification', ''), 400)}")
                out.append("")
                out.append(_truncate(p.get("detail", "")))
                out.append("")

    if len(out) <= 3:
        out.append("_No rounds have completed yet; this is the first experiment of the cell._")

    return "\n".join(out) + "\n"
