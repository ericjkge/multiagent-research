"""Research progress as a function of compute spent.

The x-axis that matters is training runs (equivalently GPU seconds), not
rounds: cells with different agent counts consume runs at different rates per
round, and compute is the resource being held fixed across the grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .common import load_json, load_records, of_type


def trajectory(run_dir: Path) -> list[dict]:
    """One row per training run, in the order the GPU executed them."""
    records = load_records(run_dir)
    starts = {r["round"]: r for r in of_type(records, "round_start")}
    candidates = sorted(
        of_type(records, "candidate"), key=lambda c: (c["round"], c["agent"], c["variant"])
    )

    rows: list[dict] = []
    best = None
    for i, c in enumerate(candidates, start=1):
        baseline = starts.get(c["round"], {}).get("baseline_bpb")
        if best is None:
            best = baseline
        if c.get("status") == "ok" and c.get("val_bpb") is not None:
            best = min(best, c["val_bpb"]) if best is not None else c["val_bpb"]
        rows.append(
            {
                "run_index": i,
                "round": c["round"],
                "agent": c["agent"],
                "variant": c["variant"],
                "val_bpb": c.get("val_bpb"),
                "status": c.get("status"),
                "best_so_far": best,
                "cost_usd": c.get("cost_usd", 0.0),
            }
        )
    return rows


def to_tsv(rows: list[dict]) -> str:
    cols = ["run_index", "round", "agent", "variant", "status", "val_bpb", "best_so_far", "cost_usd"]
    out = ["\t".join(cols)]
    for r in rows:
        out.append("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols))
    return "\n".join(out) + "\n"


def main() -> int:
    run_dir = Path(sys.argv[1])
    rows = trajectory(run_dir)
    (run_dir / "trajectory.tsv").write_text(to_tsv(rows))

    summary = load_json(run_dir, "summary.json")
    start = rows[0]["best_so_far"] if rows else None
    end = rows[-1]["best_so_far"] if rows else None
    print(f"{run_dir.name}: {len(rows)} runs, "
          f"val_bpb {start:.6f} -> {end:.6f} "
          f"({(start - end):.6f} absolute, {100 * (start - end) / start:.2f}%)"
          if start and end else "no runs")
    print(f"wrote {run_dir / 'trajectory.tsv'}")
    if summary:
        print(f"agent cost ${summary.get('agent_cost_usd', 0):.2f}, "
              f"{summary.get('training_runs', 0)} training runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
