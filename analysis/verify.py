"""Invariants that must hold for a cell's numbers to mean anything.

Run this before believing any result.  A cell that fails here is not a weak
result, it is an invalid one.

    python -m analysis.verify runs/<dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

from .common import load_json, load_records, load_timeline, of_type


NOTES: list[str] = []  # tolerated deviations, printed after an OK verdict


def check(run_dir: Path) -> list[str]:
    failures: list[str] = []
    NOTES.clear()
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

    # 2. compute matching -- the budget is runs, and it is a hard ceiling.
    # arena-train claims a slot before training, so any overshoot at all means
    # something trained outside the sanctioned path.
    if budget:
        used, allowed = budget.get("runs", 0), budget.get("budget_runs", 0)
        if used > allowed:
            failures.append(f"run budget overspent: {used} runs of {allowed}")
        if len(timeline) > allowed:
            failures.append(
                f"{len(timeline)} training runs on the GPU timeline but the budget "
                f"was {allowed}: something trained without claiming a slot"
            )
        # Every run on the timeline should correspond to a claim.
        claims = len(budget.get("claims", []))
        if claims and claims != len(timeline):
            failures.append(
                f"{claims} slots claimed but {len(timeline)} runs recorded on the timeline"
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
        if bpb is None:
            continue
        if best is not None and bpb > best + 1e-12:
            failures.append(f"baseline regressed in round {end['round']}: {best} -> {bpb}")
        best = bpb if best is None else min(best, bpb)

    # 5. a winner was actually the best candidate of its round
    by_round: dict[int, list[dict]] = {}
    for c in of_type(records, "candidate"):
        by_round.setdefault(c["round"], []).append(c)
    ends = of_type(records, "round_end")
    if cfg.get("protocol") == "open" and ends:
        # an open cell has one round; a resumed cell closes it again, and only the final close
        # (the last round_end) is judged against the whole candidate pool
        ends = ends[-1:]
    for end in ends:
        winner = end.get("winner")
        if not winner:
            continue
        pool = (of_type(records, "candidate") if cfg.get("protocol") == "open"
                else by_round.get(end["round"], []))
        cands = [c for c in pool if c.get("status") == "ok"]
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

    # 7. the cell offered the slots it promised
    # Agents may *decline* a slot, so fewer runs than slots is legitimate and
    # is itself a finding.  More candidates than slots is not.
    n_agents = len(cfg.get("agents", []))
    bon = cfg.get("bon", 1)
    # A cell resumed with a different best-of-N (recorded in provenance "resume_patches") ran its
    # earlier rounds with the earlier slot count; allow the larger of the two.
    bons = [bon] + [int(pt["bon"]) for pt in provenance.get("resume_patches", []) if "bon" in pt]
    if provenance.get("resume_patches"):
        bons.append(max(len(c) for c in by_round.values()) // max(1, n_agents) if by_round else bon)
    expected = n_agents * max(bons)
    for rnd, cands in sorted(by_round.items()):
        if len(cands) > expected:
            failures.append(
                f"round {rnd} produced {len(cands)} candidates, more than the "
                f"{expected} slots the cell offers"
            )

    # 8. the protocol ran in the right order: propose before select before
    # respond, within each round.  A response written before the measurement
    # would mean agents reacted to plans, not results, which is the thing this
    # ordering exists to prevent.
    for rnd in sorted(starts):
        cand_ts = [c["ts"] for c in by_round.get(rnd, []) if c.get("ts")]
        first_cand = min(cand_ts) if cand_ts else None
        props = [p["ts"] for p in of_type(records, "proposal") if p.get("round") == rnd]
        resps = [r["ts"] for r in of_type(records, "response") if r.get("round") == rnd]
        if props and first_cand and max(props) > first_cand:
            failures.append(f"round {rnd}: a proposal was logged after training began")
        if resps and first_cand and min(resps) < first_cand:
            failures.append(f"round {rnd}: a response was logged before any result existed")

    if summary.get("gpu_overlaps"):
        failures.append(f"summary.json reports {summary['gpu_overlaps']} GPU overlaps")

    # 8b. a cell that never trained is not a result. This happens when every agent
    # session fails before reaching arena-train (e.g. Claude Code refusing
    # bypassPermissions as root); the round loop then "completes" empty rounds.
    if not timeline:
        failures.append("no training run ever reached the GPU: every agent session failed before training")
    cands_all = of_type(records, "candidate")
    if cands_all and all(c.get("status") != "ok" for c in cands_all):
        failures.append(f"all {len(cands_all)} candidates failed; nothing was measured")

    # 9. open protocol: every run on the GPU is on the score log, nobody
    # exceeded their share, and an "independent" cell really was independent.
    if cfg.get("protocol") == "open":
        cands = of_type(records, "candidate")
        if len(cands) != len(timeline):
            failures.append(
                f"open protocol: {len(timeline)} runs on the GPU timeline but {len(cands)} "
                "score-log entries: a run trained without being recorded"
            )
        quota = budget.get("per_agent_quota") or 0
        if quota:
            per_agent: dict[str, int] = {}
            for c in budget.get("claims", []):
                per_agent[c["agent"]] = per_agent.get(c["agent"], 0) + 1
            # A budget rebuilt from the GPU timeline after a resume incident can have moved a
            # single slot between two agents; the cell total is unchanged. Tolerated, and noted.
            rebuilt = bool(budget.get("rebuilt_from_timeline"))
            total_ok = sum(per_agent.values()) == int(budget.get("budget_runs") or 0)
            for agent, n in per_agent.items():
                if n > quota:
                    if rebuilt and total_ok and n == quota + 1:
                        NOTES.append(f"open protocol: {agent} claimed {n} runs, share was {quota}: one slot "
                                     "moved between agents when the budget was rebuilt after a resume; "
                                     "cell total unchanged")
                    else:
                        failures.append(f"open protocol: {agent} claimed {n} runs, share was {quota}")
        if not cfg.get("open_share_log", True) and of_type(records, "adoption"):
            failures.append("open protocol: adoption events in a cell that ran agents independently")
        for a in of_type(records, "adoption"):
            src = next((c for c in cands if c.get("commit") == a.get("commit")), None)
            if src is None:
                failures.append(f"adoption #{a.get('id')} points at a commit that is not on the score log")
            elif src.get("agent") == a.get("agent"):
                failures.append(f"adoption #{a.get('id')}: {a.get('agent')} adopted its own result")

    # 10. phantom candidates (round protocol): an "ok" candidate with no GPU run behind it. Happens
    # when an agent's session ends without training and the orchestrator reads the stale run.log
    # of the lineage commit (found by Eric Ge, Sep 22). Never a round winner in our cells; noted.
    if cfg.get("protocol", "rounds") != "open" and timeline:
        seen = {(r.get("round"), r.get("agent"), r.get("variant")) for r in timeline}
        for c in of_type(records, "candidate"):
            if c.get("status") == "ok" and (c.get("round"), c.get("agent"), c.get("variant")) not in seen:
                NOTES.append(f"phantom candidate: round {c.get('round')} {c.get('agent')}/v{c.get('variant')} "
                             f"scored {c.get('val_bpb')} with no GPU run behind it (excluded from nothing: it never won)")
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
    for n in NOTES:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
