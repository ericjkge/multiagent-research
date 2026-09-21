"""The grid, side by side -- the headline output of the experiment.

Every other analysis module looks at one cell.  This one puts cells next to
each other, which is the only view that answers the actual question: does
adding agents, or diversifying them, buy you research progress per unit of
compute?

    python -m analysis.compare runs/haiku_1 runs/haiku_3 runs/haiku_6
    python -m analysis.compare runs/haiku_* --plot fig_haiku.png

The x-axis of the plot is **training runs consumed**, not rounds.  Compute is
what is held fixed across the grid; rounds are not comparable between cells
because a 6-agent round costs six runs and a 1-agent round costs one.
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from .common import load_json, load_records, of_type
from .curves import trajectory
from .diversity import within_round


def summarise(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    summary = load_json(run_dir, "summary.json")
    provenance = load_json(run_dir, "provenance.json")
    cfg = provenance.get("cell", {})
    records = load_records(run_dir)
    rows = trajectory(run_dir)

    starts = of_type(records, "round_start")
    baseline = starts[0].get("baseline_bpb") if starts else None
    final = summary.get("final_val_bpb")

    candidates = of_type(records, "candidate")
    crashes = sum(1 for c in candidates if c.get("status") != "ok")

    div = [r["mean_similarity"] for r in within_round(run_dir)
           if r.get("mean_similarity") is not None]

    return {
        "cell": summary.get("cell_id", run_dir.name),
        "dir": run_dir,
        "agents": len(cfg.get("agents", [])),
        "bon": cfg.get("bon", 1),
        "baseline": baseline,
        "final": final,
        "gain": (baseline - final) if (baseline is not None and final is not None) else None,
        "runs": summary.get("training_runs", len(candidates)),
        "rounds": summary.get("rounds_completed", 0),
        "crash_rate": crashes / len(candidates) if candidates else None,
        "mean_similarity": st.mean(div) if div else None,
        "agent_cost_usd": summary.get("agent_cost_usd"),
        "rows": rows,
    }


def table(cells: list[dict]) -> str:
    head = (f"{'cell':<14}{'agents':>7}{'BoN':>5}{'baseline':>11}{'final':>11}"
            f"{'gain':>10}{'runs':>6}{'rounds':>8}{'crash%':>8}{'simil.':>8}{'agent $':>9}")
    out = [head, "-" * len(head)]
    for c in cells:
        out.append(
            f"{c['cell']:<14}{c['agents']:>7}{c['bon']:>5}"
            f"{(c['baseline'] or 0):>11.6f}{(c['final'] or 0):>11.6f}"
            f"{(c['gain'] or 0):>10.6f}{c['runs']:>6}{c['rounds']:>8}"
            f"{100 * (c['crash_rate'] or 0):>8.1f}"
            f"{(c['mean_similarity'] or 0):>8.3f}{(c['agent_cost_usd'] or 0):>9.2f}"
        )
    return "\n".join(out)


def plot(cells: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    for c in cells:
        rows = c["rows"]
        if rows:
            left.step([r["run_index"] for r in rows],
                      [r["best_so_far"] for r in rows],
                      where="post", marker="o", ms=3, label=c["cell"])
        div = within_round(c["dir"])
        if div:
            right.plot([d["round"] for d in div],
                       [d.get("mean_similarity") for d in div],
                       marker="o", ms=3, label=c["cell"])

    left.set_xlabel("training runs consumed")
    left.set_ylabel("best val_bpb so far")
    left.set_title("Research progress per unit of compute")
    left.legend(fontsize=8)

    right.set_xlabel("round")
    right.set_ylabel("mean pairwise proposal similarity")
    right.set_title("Diversity collapse")
    right.set_ylim(0, 1)
    right.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--plot", type=Path, help="write a two-panel figure here")
    args = ap.parse_args()

    cells = [summarise(r) for r in args.runs]
    print(table(cells))

    if args.plot:
        plot(cells, args.plot)

    unverified = [c["cell"] for c in cells if not c["final"]]
    if unverified:
        print(f"\n!! no final metric for: {', '.join(unverified)} — run analysis.verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
