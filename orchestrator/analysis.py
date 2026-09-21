#!/usr/bin/env python3
"""Summarise one or more cells: best-so-far curves, per-round stats, diversity, crash rates, prediction error.

Usage: python orchestrator/analysis.py runs/haiku_1 runs/haiku_3 runs/haiku_6 [--plot out.png]
Diversity per round = 1 - mean pairwise Jaccard similarity of the proposals' word sets (1 = all different, 0 = identical).
"""
import argparse
import itertools
import json
import os
import re
import statistics as st

STOP = set("the a an to of and in for with on by from is it this that as at be we use using set change".split())


def words(s):
    return {w for w in re.findall(r"[a-z_]+[0-9.]*", (s or "").lower()) if w not in STOP}


def load(run):
    rows = [json.loads(l) for l in open(os.path.join(run, "results.jsonl")) if l.strip()]
    return rows


def summarise(run):
    rows = load(run)
    baseline = next((r["val_bpb"] for r in rows if r["agent"] == "baseline"), None)
    rounds = sorted({r["round"] for r in rows if r["round"] > 0})
    best_curve, div_curve, crash_curve, pred_err = [], [], [], []
    best = baseline
    for rd in rounds:
        rr = [r for r in rows if r["round"] == rd]
        vals = [r["val_bpb"] for r in rr if r.get("val_bpb") is not None]
        if vals:
            best = min(best, min(vals)) if best is not None else min(vals)
        best_curve.append(best)
        crash_curve.append(sum(1 for r in rr if r.get("val_bpb") is None) / max(1, len(rr)))
        ideas = [r.get("proposal") or r.get("idea") for r in rr]
        sims = [len(words(a) & words(b)) / max(1, len(words(a) | words(b))) for a, b in itertools.combinations(ideas, 2)]
        div_curve.append(1 - st.mean(sims) if sims else None)
        for r in rr:
            if r.get("predicted_val_bpb") is not None and r.get("val_bpb") is not None:
                try:
                    pred_err.append(abs(float(r["predicted_val_bpb"]) - r["val_bpb"]))
                except (TypeError, ValueError):
                    pass
    n_runs = sum(1 for r in rows if r["round"] > 0)
    return {"run": run, "baseline": baseline, "final_best": best, "gain": (baseline - best) if baseline and best else None,
            "runs": n_runs, "rounds": len(rounds), "best_curve": best_curve, "diversity_curve": div_curve,
            "crash_rate": st.mean(crash_curve) if crash_curve else None,
            "mean_abs_pred_err": st.mean(pred_err) if pred_err else None}


def cost(run):
    p = os.path.join(run, "agent_calls.jsonl")
    if not os.path.exists(p):
        return None
    tot = 0.0
    for l in open(p):
        try:
            c = json.loads(l)["meta"].get("cost_usd")
            tot += c or 0
        except Exception:
            pass
    return round(tot, 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--plot")
    a = ap.parse_args()
    sums = [summarise(r) for r in a.runs]
    print(f"{'cell':22s} {'baseline':>9s} {'best':>9s} {'gain':>8s} {'runs':>5s} {'crash%':>7s} {'pred err':>9s} {'agent $':>8s}")
    for s in sums:
        print(f"{os.path.basename(s['run']):22s} {s['baseline'] or 0:9.5f} {s['final_best'] or 0:9.5f} "
              f"{(s['gain'] or 0):8.5f} {s['runs']:5d} {100 * (s['crash_rate'] or 0):7.1f} "
              f"{(s['mean_abs_pred_err'] or 0):9.4f} {cost(s['run']) or 0:8.2f}")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for s in sums:
            ax[0].plot(range(1, len(s["best_curve"]) + 1), s["best_curve"], marker="o", label=os.path.basename(s["run"]))
            ax[1].plot(range(1, len(s["diversity_curve"]) + 1), [d if d is not None else float("nan") for d in s["diversity_curve"]],
                       marker="o", label=os.path.basename(s["run"]))
        ax[0].set_xlabel("round"); ax[0].set_ylabel("best val_bpb so far"); ax[0].legend()
        ax[1].set_xlabel("round"); ax[1].set_ylabel("proposal diversity (1 - Jaccard)"); ax[1].set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)
        print("wrote", a.plot)
