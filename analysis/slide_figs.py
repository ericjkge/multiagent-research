"""Five slide figures for the Sep 24 talk. Run: .venv/bin/python -m analysis.slide_figs"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "analysis_out" / "figs"

# reference categorical palette, fixed order; greys for text and scaffolding
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
OPEN_C, ROUNDS_C = BLUE, ORANGE

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "legend.frameon": False, "lines.linewidth": 2,
})


def summary(cell):
    return json.loads((RES / cell / "summary.json").read_text())


def baseline(cell):
    for line in open(RES / cell / "log.jsonl"):
        e = json.loads(line)
        if "baseline_bpb" in e:
            return e["baseline_bpb"]
    raise ValueError(cell)


def gain(cell):
    return baseline(cell) - summary(cell)["final_val_bpb"]


def curve(cell):
    """best-so-far val_bpb by run index, starting at the baseline (run 0)."""
    xs, ys = [0], [baseline(cell)]
    with open(RES / cell / "trajectory.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            xs.append(int(r["run_index"]))
            ys.append(min(ys[-1], float(r["best_so_far"])))
    return xs, ys


def crash_pct(cell):
    with open(RES / cell / "trajectory.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return 100 * sum(r["status"] != "ok" for r in rows) / len(rows)


def title(fig, head, sub):
    fig.text(0.02, 0.965, head, fontsize=18, weight="bold", color=INK, va="top")
    fig.text(0.02, 0.905, sub, fontsize=12.5, color=INK2, va="top")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


# 1. model strength is the largest lever (isolated, one box) -------------------
def fig_models():
    cells = [("Opus", "indep_opus_6_iso"), ("Sonnet", "indep_sonnet_6_iso"), ("Haiku", "indep_haiku_6_iso")]
    fig, ax = plt.subplots(figsize=(10, 5.625))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.13)
    title(fig, "Model strength is the biggest lever",
          "Six isolated agents per model, 36 five-minute runs, same GPU box (pod 4, baseline 1.0123)")
    gains = [gain(c) for _, c in cells]
    xs = range(len(cells))
    ax.bar(xs, gains, width=0.55, color=BLUE, zorder=2)
    for x, (name, c), g in zip(xs, cells, gains):
        ax.text(x, g + 0.0006, f"{g:.3f}", ha="center", va="bottom", fontsize=15, weight="bold", color=INK)
        ax.text(x, g / 2, f"{crash_pct(c):.0f}% of runs\ncrashed", ha="center", va="center",
                fontsize=11.5, color="white")
    ax.set_xticks(list(xs), [n for n, _ in cells], fontsize=14)
    ax.set_ylabel("gain in val_bpb (baseline − final)")
    ax.set_ylim(0, max(gains) * 1.2)
    ax.grid(axis="x", visible=False)
    fig.text(0.1, 0.03, "One seed per model. Seed-to-seed spread of a whole cell is 0.0015–0.0024.",
             fontsize=10.5, color=MUTED)
    save(fig, "fig1_model_strength.png")


# 2. headcount: six beat one for strong models only -----------------------------
def fig_headcount():
    panels = [
        ("Opus  (Monday box)", [("open", ["open_opus_1", "open_opus_3", "open_opus_6"]),
                                ("rounds", ["opus_1", "opus_3", "opus_6"])]),
        ("Sonnet  (Eric's box, rounds · Riddhi's box, open)",
         [("open", ["open_sonnet_1", "open_sonnet_3", None]), ("rounds", ["sonnet_1", "sonnet_3", "sonnet_6"])]),
        ("Haiku  (Anthony's box)", [("open", ["open_haiku_1", "open_haiku_3", "open_haiku_6"]),
                                    ("rounds", ["haiku_1", "haiku_3", "haiku_6"])]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 5.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.19, wspace=0.12)
    title(fig, "More agents help only a model that can use the extra tries",
          "Gain at 36 runs vs number of agents sharing the budget. Compare within a panel; boxes differ.")
    ns = [1, 3, 6]
    for ax, (name, series) in zip(axes, panels):
        for proto, cells in series:
            color = OPEN_C if proto == "open" else ROUNDS_C
            pts = [(n, gain(c)) for n, c in zip(ns, cells) if c]
            ax.plot(*zip(*pts), color=color, marker="o", markersize=8, zorder=3,
                    markeredgecolor=SURF, markeredgewidth=2)
            n_last, g_last = pts[-1]
            if name.startswith("Haiku"):
                g_last += 0.0012 if proto == "rounds" else -0.0010
            ax.text(n_last + 0.25, g_last, proto, color=INK2, va="center", fontsize=12)
        if name.startswith("Opus"):
            g2 = gain("open_opus_6_s2")
            ax.plot([6], [g2], marker="o", markersize=8, markerfacecolor=SURF, markeredgecolor=OPEN_C,
                    markeredgewidth=2, linestyle="none", zorder=3)
            ax.text(5.75, g2, "seed 2", color=MUTED, ha="right", va="center", fontsize=10.5)
        ax.set_title(name, loc="left", color=INK, fontsize=12.5)
        ax.set_xticks(ns)
        ax.set_xlim(0.5, 7.4)
        ax.set_xlabel("agents")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("gain in val_bpb")
    axes[0].set_ylim(0, 0.04)
    handles = [plt.Line2D([], [], color=OPEN_C, marker="o", label="open (long sessions, shared findings)"),
               plt.Line2D([], [], color=ROUNDS_C, marker="o", label="rounds (blind proposals, one winner)")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.9), ncol=2, fontsize=11.5)
    fig.text(0.07, 0.02, "One seed per point except Opus open 6. Haiku rounds stopped at 20–28 runs.",
             fontsize=10.5, color=MUTED)
    save(fig, "fig2_headcount.png")


# 3. workflow: open beats rounds at every headcount ------------------------------
def fig_workflow():
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 5.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.19, wspace=0.08)
    title(fig, "Workflow matters as much as headcount",
          "Opus, Monday box: best val_bpb found so far vs training runs spent (lower is better)")
    for ax, n in zip(axes, [1, 3, 6]):
        base = None
        for proto, cell, color in [("open", f"open_opus_{n}", OPEN_C), ("rounds", f"opus_{n}", ROUNDS_C)]:
            xs, ys = curve(cell)
            base = ys[0]
            ax.step(xs, ys, where="post", color=color, zorder=3)
            ax.text(xs[-1] + 0.6, ys[-1], f"{proto}\n{ys[-1]:.4f}", color=INK2, va="center", fontsize=11)
        ax.axhline(base, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(f"{n} agent{'s' if n > 1 else ''}", loc="left", color=INK)
        ax.set_xlim(0, 44)
        ax.set_xticks([0, 12, 24, 36])
        ax.set_xlabel("training runs")
    axes[0].text(1, base + 0.0004, "baseline", color=MUTED, fontsize=10.5)
    axes[0].set_ylabel("best val_bpb so far")
    axes[0].set_ylim(0.974, 0.999)
    fig.text(0.07, 0.02, "Rounds spent 8x the agent tokens of open for a worse score. Noise per run: 0.0008.",
             fontsize=10.5, color=MUTED)
    save(fig, "fig3_workflow.png")


# 4. sharing: not detectable against a truly isolated control --------------------
def fig_sharing():
    fig, ax = plt.subplots(figsize=(10, 5.625))
    fig.subplots_adjust(left=0.1, right=0.76, top=0.78, bottom=0.17)
    title(fig, "Sharing findings bought nothing detectable",
          "Six Opus agents, 36 runs, pod 4: shared log vs no access to each other at all")
    for cell, label, color in [("open_opus_6_iso", "shared log", OPEN_C),
                               ("indep_opus_6_iso", "isolated", GREEN)]:
        xs, ys = curve(cell)
        ax.step(xs, ys, where="post", color=color, zorder=3)
        ax.text(37, ys[-1], f"{label}  {ys[-1]:.4f}", color=INK2, va="center", fontsize=12)
    base = curve("open_opus_6_iso")[1][0]
    ax.axhline(base, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.5, base + 0.0006, "baseline", color=MUTED, fontsize=10.5)
    # detection threshold bracket centred on the two finals
    a, b = summary("open_opus_6_iso")["final_val_bpb"], summary("indep_opus_6_iso")["final_val_bpb"]
    mid = (a + b) / 2
    ax.add_patch(Rectangle((33.2, mid - 0.003), 2.3, 0.006, color=GRID, zorder=1))
    fig.text(0.775, 0.42, f"Δ = {abs(a - b):.4f}\ngrey bar: smallest\ndifference one seed\nper arm can detect\n(≈ 0.006)",
             fontsize=10.5, color=INK2, va="bottom")
    ax.set_xlim(0, 36)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
    ax.set_ylim(0.978, 1.016)
    ax.set_xlabel("training runs")
    ax.set_ylabel("best val_bpb so far")
    fig.text(0.1, 0.02, "Monday's 'independent' cells read the shared score table; these isolated reruns replace them.",
             fontsize=10.5, color=MUTED)
    save(fig, "fig4_sharing.png")


# 5. herding: blind rounds converge on the same idea -----------------------------
IDEAS = [  # (substring in proposal title, short label, colour or None for one-offs)
    ("U-net", "U-net skips", None), ("Depth 12", "depth 12 + HD64", None), ("Shrink width", "narrower", None),
    ("atch", "halve batch", BLUE), ("HEAD_DIM", "HEAD_DIM 64", ORANGE), ("armdown", "longer warmdown", AQUA),
    ("window", "window 512", VIOLET), ("DEVICE_BATCH", "device batch 128", MAGENTA),
    ("RoPE", "RoPE base", None), ("autotune", "max-autotune", YELLOW), ("fp32", "bf16 logits", None),
    ("Share one", "shared VE table", None), ("every layer", "VE all layers", None),
]


def idea(t):
    for key, label, color in IDEAS:
        if key == "atch" and "DEVICE_BATCH" in t:
            continue
        if key.lower() in t.lower():
            return label, color
    return t[:18], None


def fig_herding():
    props, winners = {}, {}
    for line in open(RES / "opus_6" / "log.jsonl"):
        e = json.loads(line)
        if e["t"] == "proposal":
            props[(e["round"], e["agent"])] = e["title"]
        elif e["t"] == "round_end" and e["winner"]:
            winners[e["round"]] = e["winner"]["agent"]
    rounds = sorted({r for r, _ in props})
    agents = sorted({a for _, a in props})
    fig, ax = plt.subplots(figsize=(13.33, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.12)
    title(fig, "Blind rounds herd: six agents, one idea",
          "Opus, 6 agents, rounds protocol. Each cell is one agent's proposal; no agent could see the others'.")
    for i, r in enumerate(rounds):
        labels = [idea(props[(r, a)]) for a in agents]
        for j, (a, (label, color)) in enumerate(zip(agents, labels)):
            fc = color or "#f0efea"
            ax.add_patch(Rectangle((i + 0.04, j + 0.06), 0.92, 0.88, facecolor=fc, edgecolor="none"))
            if winners.get(r) == a:
                ax.add_patch(Rectangle((i + 0.04, j + 0.06), 0.92, 0.88, facecolor="none",
                                       edgecolor=INK, linewidth=2.5))
            ax.text(i + 0.5, j + 0.5, label, ha="center", va="center", fontsize=10.5,
                    color="white" if color in (BLUE, ORANGE, VIOLET) else INK)
        distinct = len({l for l, _ in labels})
        ax.text(i + 0.5, len(agents) + 0.35, f"{distinct} distinct idea{'s' if distinct > 1 else ''}",
                ha="center", fontsize=11.5, color=INK2, weight="bold" if distinct <= 2 else "normal")
    ax.set_xlim(0, len(rounds))
    ax.set_ylim(len(agents) + 0.8, 0)
    ax.set_xticks([i + 0.5 for i in range(len(rounds))], [f"round {r}" for r in rounds])
    ax.set_yticks([j + 0.5 for j in range(len(agents))], agents)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text(0.07, 0.015, "Black outline = round winner. Rounds 1–2: all 12 proposals contained HEAD_DIM 64 and none beat the baseline, "
             "so neither round had a winner.", fontsize=10.5, color=MUTED)
    save(fig, "fig5_herding.png")


if __name__ == "__main__":
    fig_models()
    fig_headcount()
    fig_workflow()
    fig_sharing()
    fig_herding()
