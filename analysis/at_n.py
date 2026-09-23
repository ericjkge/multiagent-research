"""Best val_bpb reached by run n, for every cell: the equal-compute comparison.

Cells that stopped short of the budget (a round-based cell cannot add a partial
round after lost slots) are compared with the others at the same run count.

    python -m analysis.at_n results/* --n 34 36
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import load_json
from .curves import trajectory


def best_at(rows: list[dict], n: int):
    best = None
    for r in rows:
        if r["run_index"] > n:
            break
        if r.get("best_so_far") is not None:
            best = r["best_so_far"]
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--n", nargs="+", type=int, default=[34, 36])
    a = ap.parse_args()
    print("| cell | runs | " + " | ".join(f"best@{n}" for n in a.n) + " | final |")
    print("|---|---:|" + "---:|" * (len(a.n) + 1))
    for d in a.dirs:
        d = Path(d)
        s = load_json(d, "summary.json")
        rows = trajectory(d)
        vals = []
        for n in a.n:
            b = best_at(rows, n)
            vals.append(f"{b:.6f}" if b is not None else "")
        print(f"| {s.get('cell_id', d.name)} | {s.get('training_runs')} | " + " | ".join(vals)
              + f" | {s.get('final_val_bpb')} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
