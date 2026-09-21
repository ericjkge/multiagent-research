"""Do errors propagate as agents are added?

Three things are measured:

``crash_rate``
    the share of candidates that never produced a ``val_bpb``.

``contagion``
    for each crashed experiment, how closely the *next* round's proposals by
    **other** agents resemble it.  A failed idea travelling through the group
    is the propagation the experiment is looking for; the shared log is the
    channel it would travel through.

``repair``
    whether agents recovered their own crashes within the attempt limit, and
    how much GPU time crashes consumed.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .common import load_records, load_timeline, of_type
from .diversity import _vectorize, proposal_text


def crash_stats(run_dir: Path) -> dict:
    records = load_records(run_dir)
    candidates = of_type(records, "candidate")
    if not candidates:
        return {}

    per_round: dict[int, Counter] = {}
    per_agent: dict[str, Counter] = {}
    for c in candidates:
        per_round.setdefault(c["round"], Counter())[c.get("status", "crash")] += 1
        per_agent.setdefault(c["agent"], Counter())[c.get("status", "crash")] += 1

    failed = sum(1 for c in candidates if c.get("status") != "ok")
    timeline = load_timeline(run_dir)
    wasted = sum(
        r["charged_s"]
        for r in timeline
        if not r.get("produced_metric", r.get("exit_code") == 0)
    )

    return {
        "n_candidates": len(candidates),
        "n_failed": failed,
        "crash_rate": round(failed / len(candidates), 4),
        "per_round": {k: dict(v) for k, v in sorted(per_round.items())},
        "per_agent": {k: dict(v) for k, v in sorted(per_agent.items())},
        "gpu_seconds_on_failed_runs": round(wasted, 1),
    }


def contagion(run_dir: Path, threshold: float = 0.25) -> list[dict]:
    """Crashed ideas that reappear in another agent's next proposal."""
    from sklearn.metrics.pairwise import cosine_similarity

    records = load_records(run_dir)
    crashes: dict[int, list[dict]] = {}
    for c in of_type(records, "candidate"):
        if c.get("status") != "ok":
            crashes.setdefault(c["round"], []).append(c)

    proposals: dict[int, list[dict]] = {}
    for p in of_type(records, "proposal"):
        proposals.setdefault(p["round"], []).append(p)

    rows = []
    for rnd, failed in sorted(crashes.items()):
        following = proposals.get(rnd + 1, [])
        if not following:
            continue
        for c in failed:
            crashed_text = f"{c.get('title', '')} {c.get('description', '')} {c.get('reflection', '')}"
            others = [p for p in following if p["agent"] != c["agent"]]
            if not others or not crashed_text.strip():
                continue
            texts = [crashed_text] + [proposal_text(p) for p in others]
            sims = cosine_similarity(_vectorize(texts))[0, 1:]
            for p, s in zip(others, sims):
                if s >= threshold:
                    rows.append(
                        {
                            "crash_round": rnd,
                            "crashed_agent": c["agent"],
                            "crashed_idea": c.get("title") or c.get("description", ""),
                            "picked_up_by": p["agent"],
                            "next_title": p.get("title", ""),
                            "similarity": round(float(s), 4),
                        }
                    )
    return sorted(rows, key=lambda r: -r["similarity"])


def repair_stats(run_dir: Path) -> dict:
    timeline = load_timeline(run_dir)
    by_slot: dict[tuple, list[dict]] = {}
    for r in timeline:
        by_slot.setdefault((r["round"], r["agent"], r["variant"]), []).append(r)

    retried = {k: v for k, v in by_slot.items() if len(v) > 1}
    recovered = sum(
        1 for v in retried.values()
        if v[-1].get("produced_metric", v[-1].get("exit_code") == 0)
    )
    return {
        "slots_that_retried": len(retried),
        "retries_that_recovered": recovered,
        "mean_attempts": round(
            sum(len(v) for v in by_slot.values()) / len(by_slot), 2
        ) if by_slot else 0,
    }


def main() -> int:
    run_dir = Path(sys.argv[1])
    result = {
        "crashes": crash_stats(run_dir),
        "contagion": contagion(run_dir),
        "repair": repair_stats(run_dir),
    }
    (run_dir / "errors.json").write_text(json.dumps(result, indent=2))

    crashes = result["crashes"]
    if crashes:
        print(f"crash rate: {crashes['crash_rate']:.1%} "
              f"({crashes['n_failed']}/{crashes['n_candidates']}), "
              f"{crashes['gpu_seconds_on_failed_runs']:.0f}s of GPU on failed runs")
    print(f"repair: {result['repair']}")
    if result["contagion"]:
        print(f"\n{len(result['contagion'])} crashed idea(s) echoed by another agent next round:")
        for row in result["contagion"][:10]:
            print(f"  r{row['crash_round']} {row['crashed_agent']} \"{row['crashed_idea'][:40]}\""
                  f" -> {row['picked_up_by']} \"{row['next_title'][:40]}\" "
                  f"(sim {row['similarity']})")
    else:
        print("\nno crashed idea was echoed by another agent in the next round")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
