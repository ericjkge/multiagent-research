"""Invariants that must hold for a cell's numbers to mean anything.

Run this before believing any result.  A cell that fails here is not a weak
result, it is an invalid one.

    python -m analysis.verify runs/<dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from .common import load_json, load_records, load_timeline, of_type


def check(run_dir: Path) -> list[str]:
    failures: list[str] = []
    records = load_records(run_dir)
    summary = load_json(run_dir, "summary.json")
    budget = load_json(run_dir, "budget.json")
    timeline = load_timeline(run_dir)
    provenance = load_json(run_dir, "provenance.json")
    cfg = provenance.get("cell", {})

    # 1. the single-GPU assumption
    ordered = sorted(timeline, key=lambda r: r["started_at"])
    for earlier, later in zip(ordered, ordered[1:]):
        if later["started_at"] < earlier["ended_at"] - 1.0:
            failures.append(
                f"GPU runs overlapped: {earlier['agent']}/v{earlier['variant']} and "
                f"{later['agent']}/v{later['variant']} in round {later['round']}"
            )

    # 2. compute matching
    if budget and budget.get("spent_s", 0) > budget.get("budget_s", 0) * 1.05:
        failures.append(
            f"GPU budget overspent: {budget['spent_s']:.0f}s of {budget['budget_s']:.0f}s"
        )

    # 3. every round opened and closed
    starts = {r["round"] for r in of_type(records, "round_start")}
    ends = {r["round"] for r in of_type(records, "round_end")}
    if starts != ends:
        failures.append(f"rounds started but not ended: {sorted(starts - ends)}")

    # 4. the lineage never moved backwards
    best = None
    for end in sorted(of_type(records, "round_end"), key=lambda r: r["round"]):
        bpb = end.get("new_baseline_bpb")
        if best is not None and bpb is not None and bpb > best + 1e-12:
            failures.append(f"baseline regressed in round {end['round']}: {best} -> {bpb}")
        best = bpb if best is None else min(best, bpb)

    # 5. a winner was actually the best candidate of its round
    by_round: dict[int, list[dict]] = {}
    for c in of_type(records, "candidate"):
        by_round.setdefault(c["round"], []).append(c)
    for end in of_type(records, "round_end"):
        winner = end.get("winner")
        if not winner:
            continue
        cands = [c for c in by_round.get(end["round"], []) if c.get("status") == "ok"]
        if not cands:
            failures.append(f"round {end['round']} declared a winner with no successful runs")
            continue
        best_c = min(cands, key=lambda c: c["val_bpb"])
        if (best_c["agent"], best_c["variant"]) != (winner["agent"], winner["variant"]):
            failures.append(
                f"round {end['round']} winner {winner['agent']}/v{winner['variant']} was not "
                f"the best candidate ({best_c['agent']}/v{best_c['variant']})"
            )

    # 6. worktrees cleaned up
    leftover = list((Path(run_dir) / "work").glob("*")) if (Path(run_dir) / "work").exists() else []
    if leftover:
        failures.append(f"{len(leftover)} worktree(s) were not cleaned up")

    # 7. the cell ran the candidates it promised
    n_agents = len(cfg.get("agents", []))
    bon = cfg.get("bon", 1)
    expected = n_agents * bon
    for rnd, cands in sorted(by_round.items()):
        if len(cands) != expected:
            failures.append(
                f"round {rnd} produced {len(cands)} candidates, expected {expected}"
            )

    if summary.get("gpu_overlaps"):
        failures.append(f"summary.json reports {summary['gpu_overlaps']} GPU overlaps")

    return failures


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run_dir = Path(sys.argv[1])
    failures = check(run_dir)
    if failures:
        print(f"INVALID — {len(failures)} invariant(s) violated in {run_dir}:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK — all invariants hold for {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
