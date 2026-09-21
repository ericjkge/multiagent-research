"""Does collaboration collapse the space of ideas?

Two measures, both computed from the proposals stored verbatim in the log:

``within_round_similarity``
    mean pairwise TF-IDF cosine between the proposals made in the same round.
    Rising over the cell is the signature of diversity collapse.

``adoption``
    similarity between an agent's proposal and what *other* agents proposed in
    the previous round.  Separates genuine convergence-through-debate from
    agents independently walking into the same idea.

Both are lexical and free.  ``--classify`` adds a taxonomy pass with Haiku for
a semantic view, at a cost of roughly a cent per proposal.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from .common import load_records, of_type

TAXONOMY = [
    "optimizer", "architecture", "data", "lr_schedule",
    "regularization", "precision", "batching", "initialization", "other",
]


def _vectorize(texts: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", min_df=1, sublinear_tf=True)
    return vec.fit_transform(texts)


def _cosines(matrix) -> list[float]:
    from sklearn.metrics.pairwise import cosine_similarity

    sim = cosine_similarity(matrix)
    n = sim.shape[0]
    return [sim[i, j] for i in range(n) for j in range(i + 1, n)]


def proposal_text(p: dict) -> str:
    return f"{p.get('title', '')} {p.get('justification', '')} {p.get('detail', '')}".strip()


def within_round(run_dir: Path) -> list[dict]:
    records = load_records(run_dir)
    by_round: dict[int, list[dict]] = {}
    for p in of_type(records, "proposal"):
        by_round.setdefault(p["round"], []).append(p)

    rows = []
    for rnd, props in sorted(by_round.items()):
        if len(props) < 2:
            rows.append({"round": rnd, "n": len(props), "mean_similarity": None})
            continue
        sims = _cosines(_vectorize([proposal_text(p) for p in props]))
        rows.append(
            {
                "round": rnd,
                "n": len(props),
                "mean_similarity": round(sum(sims) / len(sims), 4),
                "max_similarity": round(max(sims), 4),
            }
        )
    return rows


def adoption(run_dir: Path) -> list[dict]:
    """How much each proposal echoes the *previous* round's other agents."""
    from sklearn.metrics.pairwise import cosine_similarity

    records = load_records(run_dir)
    by_round: dict[int, list[dict]] = {}
    for p in of_type(records, "proposal"):
        by_round.setdefault(p["round"], []).append(p)

    rows = []
    for rnd in sorted(by_round):
        prev = by_round.get(rnd - 1, [])
        if not prev:
            continue
        for p in by_round[rnd]:
            others = [q for q in prev if q["agent"] != p["agent"]]
            if not others:
                continue
            texts = [proposal_text(p)] + [proposal_text(q) for q in others]
            sim = cosine_similarity(_vectorize(texts))[0, 1:]
            rows.append(
                {
                    "round": rnd,
                    "agent": p["agent"],
                    "title": p.get("title", ""),
                    "max_prev_similarity": round(float(sim.max()), 4),
                    "closest_prev_agent": others[int(sim.argmax())]["agent"],
                    "closest_prev_title": others[int(sim.argmax())].get("title", ""),
                }
            )
    return rows


def classify(run_dir: Path, model: str = "haiku") -> list[dict]:
    """Semantic view: bucket each proposal, then measure per-round entropy."""
    records = load_records(run_dir)
    proposals = of_type(records, "proposal")
    schema = {
        "type": "object",
        "properties": {"category": {"type": "string", "enum": TAXONOMY}},
        "required": ["category"],
        "additionalProperties": False,
    }

    tagged = []
    for p in proposals:
        prompt = (
            "Classify this ML training experiment proposal into exactly one category.\n\n"
            f"Title: {p.get('title', '')}\nDetail: {p.get('detail', '')[:800]}"
        )
        proc = subprocess.run(
            ["claude", "-p", "--output-format", "json", "--model", model,
             "--effort", "low", "--tools", "", "--setting-sources", "",
             "--no-session-persistence", "--json-schema", json.dumps(schema)],
            input=prompt, capture_output=True, text=True,
        )
        try:
            payload = json.loads(proc.stdout)
            category = (payload.get("structured_output") or {}).get("category", "other")
        except json.JSONDecodeError:
            category = "other"
        tagged.append({"round": p["round"], "agent": p["agent"], "category": category})

    rows = []
    by_round: dict[int, list[str]] = {}
    for t in tagged:
        by_round.setdefault(t["round"], []).append(t["category"])
    for rnd, cats in sorted(by_round.items()):
        counts = Counter(cats)
        total = sum(counts.values())
        entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        rows.append(
            {
                "round": rnd,
                "distinct_categories": len(counts),
                "entropy_bits": round(entropy, 3),
                "categories": dict(counts),
            }
        )
    return {"per_proposal": tagged, "per_round": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--classify", action="store_true",
                    help="add the LLM taxonomy pass (costs a few cents)")
    args = ap.parse_args()

    result = {"within_round": within_round(args.run_dir), "adoption": adoption(args.run_dir)}
    if args.classify:
        result["taxonomy"] = classify(args.run_dir)

    out = args.run_dir / "diversity.json"
    out.write_text(json.dumps(result, indent=2))

    print("round  n  mean_sim  max_sim")
    for r in result["within_round"]:
        ms = r.get("mean_similarity")
        xs = r.get("max_similarity")
        print(f"{r['round']:>5}  {r['n']}  "
              f"{'n/a' if ms is None else f'{ms:8.4f}'}  "
              f"{'n/a' if xs is None else f'{xs:7.4f}'}")
    if result["adoption"]:
        mean_adopt = sum(a["max_prev_similarity"] for a in result["adoption"]) / len(result["adoption"])
        print(f"\nmean similarity to the previous round's other agents: {mean_adopt:.4f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
