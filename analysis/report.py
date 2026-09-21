"""One readable report per cell.

    python -m analysis.report runs/<dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .common import load_json, load_records, of_type
from .curves import to_tsv, trajectory
from .diversity import adoption, within_round
from .errors import contagion, crash_stats, repair_stats


def build(run_dir: Path) -> str:
    summary = load_json(run_dir, "summary.json")
    provenance = load_json(run_dir, "provenance.json")
    cfg = provenance.get("cell", {})
    records = load_records(run_dir)
    rows = trajectory(run_dir)

    lines: list[str] = [f"# {summary.get('cell_id', run_dir.name)}", ""]

    lines += [
        "## Setup",
        "",
        f"- agents: {len(cfg.get('agents', []))} x BoN={cfg.get('bon')} "
        f"= {len(cfg.get('agents', [])) * cfg.get('bon', 1)} candidates/round",
        f"- models: {', '.join(sorted({a['model'] for a in cfg.get('agents', [])}))}",
        f"- effort: {', '.join(sorted({a['effort'] for a in cfg.get('agents', [])}))}",
        f"- GPU budget: {summary.get('gpu_seconds_budget', 0) / 3600:.2f} h "
        f"({cfg.get('train_run_budget')} nominal runs)",
        f"- autoresearch commit: {provenance.get('autoresearch_commit', '')[:7]}",
        f"- claude: {provenance.get('claude_version', '?')}",
        f"- config fingerprint: {provenance.get('config_fingerprint', '?')}",
        "",
    ]

    start = rows[0]["best_so_far"] if rows else None
    end = rows[-1]["best_so_far"] if rows else None
    lines += [
        "## Result",
        "",
        f"- rounds completed: {summary.get('rounds_completed')}",
        f"- training runs: {summary.get('training_runs')} "
        f"({summary.get('gpu_seconds_spent', 0) / 3600:.2f} GPU-h used)",
        f"- val_bpb: {start:.6f} -> {end:.6f}" if start and end else "- val_bpb: n/a",
        f"- improvement: {100 * (start - end) / start:.2f}%" if start and end and start else "",
        f"- agent cost: ${summary.get('agent_cost_usd', 0):.2f}",
        f"- wall clock: {summary.get('wall_clock_s', 0) / 3600:.2f} h",
        f"- stopped because: {summary.get('stop_reason', '')}",
        "",
    ]

    lines += ["## Progress per round", "", "| round | candidates | crashes | best val_bpb | winner |", "|---|---|---|---|---|"]
    for end_rec in sorted(of_type(records, "round_end"), key=lambda r: r["round"]):
        w = end_rec.get("winner")
        lines.append(
            f"| {end_rec['round']} | {end_rec.get('n_candidates', 0)} | "
            f"{end_rec.get('n_crashes', 0)} | {end_rec.get('new_baseline_bpb', 0):.6f} | "
            f"{(w['agent'] + '/v' + str(w['variant'])) if w else '—'} |"
        )
    lines.append("")

    div = within_round(run_dir)
    if div:
        lines += ["## Idea diversity", "", "| round | proposals | mean pairwise similarity |", "|---|---|---|"]
        for d in div:
            ms = d.get("mean_similarity")
            lines.append(f"| {d['round']} | {d['n']} | {'n/a' if ms is None else f'{ms:.4f}'} |")
        adopt = adoption(run_dir)
        if adopt:
            mean_adopt = sum(a["max_prev_similarity"] for a in adopt) / len(adopt)
            lines += ["", f"Mean similarity to the previous round's *other* agents: **{mean_adopt:.4f}**"]
        lines.append("")

    crashes = crash_stats(run_dir)
    if crashes:
        lines += [
            "## Errors",
            "",
            f"- crash rate: {crashes['crash_rate']:.1%} "
            f"({crashes['n_failed']}/{crashes['n_candidates']})",
            f"- GPU spent on failed runs: {crashes['gpu_seconds_on_failed_runs']:.0f}s",
            f"- repair: {repair_stats(run_dir)}",
            "",
        ]
        spread = contagion(run_dir)
        if spread:
            lines += ["Crashed ideas echoed by another agent the next round:", ""]
            for row in spread[:10]:
                lines.append(
                    f"- r{row['crash_round']} `{row['crashed_agent']}` "
                    f"\"{row['crashed_idea'][:50]}\" -> `{row['picked_up_by']}` "
                    f"\"{row['next_title'][:50]}\" (sim {row['similarity']})"
                )
        else:
            lines.append("No crashed idea was echoed by another agent in the next round.")
        lines.append("")

    per_agent: dict[str, dict] = {}
    for c in of_type(records, "candidate"):
        slot = per_agent.setdefault(c["agent"], {"runs": 0, "ok": 0, "wins": 0, "cost": 0.0})
        slot["runs"] += 1
        slot["ok"] += 1 if c.get("status") == "ok" else 0
        slot["cost"] += c.get("cost_usd", 0.0)
    for end_rec in of_type(records, "round_end"):
        w = end_rec.get("winner")
        if w and w["agent"] in per_agent:
            per_agent[w["agent"]]["wins"] += 1

    lines += ["## Per-agent attribution", "", "| agent | runs | successful | rounds won | cost |", "|---|---|---|---|---|"]
    for agent, s in sorted(per_agent.items()):
        lines.append(f"| {agent} | {s['runs']} | {s['ok']} | {s['wins']} | ${s['cost']:.2f} |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    run_dir = Path(sys.argv[1])
    (run_dir / "trajectory.tsv").write_text(to_tsv(trajectory(run_dir)))
    report = build(run_dir)
    (run_dir / "report.md").write_text(report)
    print(report)
    print(f"\nwrote {run_dir / 'report.md'} and {run_dir / 'trajectory.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
