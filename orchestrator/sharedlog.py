"""The shared log: proposals, responses, experiments and results.

This is the collaboration medium.  Agents *read* a rendered Markdown view plus
a Karpathy-format ``results.tsv``; the orchestrator owns every write to the
underlying JSONL.  That asymmetry is deliberate -- N agent processes appending
to one file concurrently would interleave and lose records, and the log is the
primary experimental artifact.  Agents keep genuine write access through the
``notes`` field of their structured output and through ``scratchpad.md``, which
is flock-guarded.

Record types: ``round_start``, ``proposal``, ``response``, ``candidate``,
``round_end``.
"""

from __future__ import annotations

import fcntl
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterable

MAX_DETAIL_CHARS = 1200


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
        record = {"t": kind, "ts": time.time(), **fields}
        with self._lock:
            with open(self.jsonl, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        return record

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

    def publish(self, upto_round: int, include_proposals_for: int | None = None) -> None:
        """Regenerate the files agents read.  Called before each phase."""
        records = self.records()
        self.markdown.write_text(
            render_log_md(records, upto_round, include_proposals_for)
        )
        self.results_tsv.write_text(render_results_tsv(records))


def _truncate(text: str, limit: int = MAX_DETAIL_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


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
) -> str:
    """The running log as the agents see it.

    Rounds strictly before ``upto_round`` are rendered in full.  The current
    round's proposals appear only during the respond phase, which is what makes
    propose independent and respond collaborative.
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
                out.append(f"- **{p['agent']} — {p.get('title', '')}**")
                out.append(f"  - why: {_truncate(p.get('justification', ''), 300)}")
                detail = _truncate(p.get("detail", ""))
                if detail:
                    out.append(f"  - detail: {detail}")
            out.append("")

        if slot.get("response"):
            out.append("### Responses")
            for resp in sorted(slot["response"], key=lambda x: x["agent"]):
                out.append(f"- **{resp['agent']}**: {_truncate(resp.get('text', ''), 900)}")
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

    if include_proposals_for is not None:
        current = by_round.get(include_proposals_for, {}).get("proposal", [])
        if current:
            out.append(f"## Round {include_proposals_for} — proposals on the table now")
            out.append("")
            for p in sorted(current, key=lambda x: x["agent"]):
                out.append(f"### {p['agent']}: {p.get('title', '')}")
                out.append(f"*Justification:* {_truncate(p.get('justification', ''), 400)}")
                out.append("")
                out.append(_truncate(p.get("detail", "")))
                out.append("")

    if len(out) <= 3:
        out.append("_No rounds have completed yet; this is the first experiment of the cell._")

    return "\n".join(out) + "\n"
