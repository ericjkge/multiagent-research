"""Five slide figures for the Sep 24 talk. Run: .venv/bin/python -m analysis.slide_figs"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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
          "Six isolated agents per model, 36 five-minute training runs each cell")
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
        ("Opus", [("open", ["open_opus_1", "open_opus_3", "open_opus_6"]),
                                ("rounds", ["opus_1", "opus_3", "opus_6"])]),
        ("Sonnet",
         [("open", ["open_sonnet_1", "open_sonnet_3", None]), ("rounds", ["sonnet_1", "sonnet_3", "sonnet_6"])]),
        ("Haiku", [("open", ["open_haiku_1", "open_haiku_3", "open_haiku_6"]),
                                    ("rounds", ["haiku_1", "haiku_3", "haiku_6"])]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.33, 5.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.74, bottom=0.19, wspace=0.12)
    title(fig, "More agents help only a model that can use the extra tries",
          "Gain at 36 training runs vs number of agents sharing that budget")
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
          "Opus: best val_bpb found so far vs training runs spent (lower is better)")
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
          "Six Opus agents, 36 runs: shared log vs no access to each other at all")
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
    fig.text(0.1, 0.02, "Isolated = own clone and own score table per agent; access to peers blocked and audited.",
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


# ---------------------------------------------------------------------------
# setup figures (diagrams) and two more results figures
# ---------------------------------------------------------------------------
PALE = "#f0efea"
VALID = [  # every cell the talk uses; Sep 22 indep_* (non-iso) cells were contaminated and are left out
    "opus_1", "opus_3", "opus_6", "open_opus_1", "open_opus_3", "open_opus_6", "open_opus_6_s2",
    "open_opus_6_iso", "indep_opus_6_iso", "sonnet_1", "sonnet_3", "sonnet_6", "open_sonnet_1",
    "open_sonnet_3", "open_sonnet_6", "indep_sonnet_6_iso", "haiku_1", "haiku_3", "haiku_6",
    "open_haiku_1", "open_haiku_3", "open_haiku_6", "indep_haiku_6_iso"]


def canvas(head, sub):
    fig = plt.figure(figsize=(13.33, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 133.3)
    ax.set_ylim(0, 75)
    ax.axis("off")
    fig.text(0.03, 0.955, head, fontsize=22, weight="bold", color=INK, va="top")
    fig.text(0.03, 0.895, sub, fontsize=14, color=INK2, va="top")
    return fig, ax


def box(ax, x, y, w, h, text, fc=PALE, ec="none", color=INK, size=13, weight="normal", lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
            weight=weight, linespacing=1.4)


def arrow(ax, p, q, color=MUTED, rad=0.0, lw=2, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=16, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))


# S1. the research loop -----------------------------------------------------
def fig_setup_loop():
    fig, ax = canvas("The task: agents doing ML research on a fixed budget",
                     "Karpathy's autoresearch: improve a small GPT's training script, judged only by the score")
    box(ax, 6, 34, 26, 16, "AI agent\n(Claude Code: Opus,\nSonnet or Haiku)", fc=BLUE, color="white", size=14,
        weight="bold")
    box(ax, 44, 34, 24, 16, "edits train.py\nany change: architecture,\nbatch, LR, schedule ...", size=13)
    box(ax, 80, 34, 24, 16, "trains for 5 minutes\non one H100", size=13)
    box(ax, 80, 8, 24, 14, "score: val_bpb\n(bits per byte, lower = better)", size=13)
    box(ax, 44, 8, 24, 14, "keep or discard,\nread the logs,\nplan the next try", size=13)
    arrow(ax, (32, 42), (44, 42))
    arrow(ax, (68, 42), (80, 42))
    arrow(ax, (92, 34), (92, 22))
    arrow(ax, (80, 15), (68, 15))
    arrow(ax, (44, 15), (19, 34), rad=-0.25)
    # budget panel
    box(ax, 110, 8, 20, 42, "", fc=SURF, ec=GRID, lw=1.5)
    ax.text(120, 45, "Budget per cell", ha="center", fontsize=13.5, weight="bold", color=INK)
    for i in range(36):
        cx, cy = 113.2 + (i % 6) * 2.75, 39 - (i // 6) * 2.75
        ax.add_patch(Rectangle((cx, cy), 2.1, 2.1, facecolor=BLUE, edgecolor="none"))
    ax.text(120, 19.5, "36 training runs\n= 3 GPU-hours", ha="center", va="center", fontsize=12.5, color=INK)
    ax.text(120, 12.5, "shared by 1, 3\nor 6 agents", ha="center", va="center", fontsize=12.5, color=INK2)
    fig.text(0.03, 0.03, "Baseline val_bpb ≈ 1.00. Run-to-run noise of an unchanged script: 0.0008.",
             fontsize=12, color=MUTED)
    save(fig, "s1_setup_loop.png")


# S2. the three ways to organise the agents -----------------------------------
def fig_setup_protocols():
    fig, ax = canvas("Three ways to organise the agents",
                     "Same 36-run budget, same models; only the organisation changes")
    cols = [(4, "Rounds", ROUNDS_C, "Everyone proposes blind, all train,\nthe best result becomes the\nshared starting point, repeat"),
            (48, "Open", OPEN_C, "Long independent sessions,\na shared log of findings and scores,\nagents may adopt each other's code"),
            (92, "Isolated (control)", GREEN, "Same as open, but no log,\nno peer code, no shared scores;\naccess blocked and audited")]
    for x0, name, color, desc in cols:
        box(ax, x0, 8, 38, 54, "", fc=SURF, ec=GRID, lw=1.5)
        ax.text(x0 + 19, 57.5, name, ha="center", fontsize=16, weight="bold", color=INK)
        ax.text(x0 + 19, 14.5, desc, ha="center", va="center", fontsize=12, color=INK2, linespacing=1.45)
    # rounds: agents -> winner -> baseline, twice
    for r, y in enumerate([44, 31]):
        for i in range(3):
            ax.add_patch(plt.Circle((10 + i * 6.5, y), 2.2, color=ROUNDS_C))
        box(ax, 29, y - 3, 10, 6, f"best of\nround {r}", size=10.5)
        arrow(ax, (25.5, y), (29, y))
    arrow(ax, (34, 41), (16.5, 34), rad=0.2)
    ax.text(26, 37.5, "new baseline for all", fontsize=10, color=MUTED)
    # open: agents around a shared log
    box(ax, 59, 33, 16, 10, "shared log\nfindings · scores\ncode to adopt", fc=PALE, size=10.5)
    for (cx, cy) in [(53, 49), (67, 51.5), (81, 49), (53, 27), (67, 24.5), (81, 27)]:
        ax.add_patch(plt.Circle((cx, cy), 2.2, color=OPEN_C))
        arrow(ax, (cx, cy), (67 + (cx - 67) * 0.35, 38 + (cy - 38) * 0.35), style="<|-|>", lw=1.4)
    # isolated: agents alone
    for (cx, cy) in [(97, 49), (111, 51.5), (125, 49), (97, 27), (111, 24.5), (125, 27)]:
        ax.add_patch(plt.Circle((cx, cy), 2.2, color=GREEN))
        ax.add_patch(Rectangle((cx - 4, cy - 4), 8, 8, facecolor="none", edgecolor=MUTED, lw=1.2, ls=(0, (3, 2))))
    ax.text(111, 38, "own clone,\nown score table", ha="center", va="center", fontsize=11, color=INK2)
    fig.text(0.03, 0.03, "Open follows Park et al. (arXiv 2609.21032). Rounds is our original lockstep design.",
             fontsize=12, color=MUTED)
    save(fig, "s2_setup_protocols.png")


# S3. the experiment matrix ---------------------------------------------------
def fig_setup_matrix():
    cols = [("rounds", 1), ("rounds", 3), ("rounds", 6), ("open", 1), ("open", 3), ("open", 6), ("isolated", 6)]
    models = ["Opus", "Sonnet", "Haiku"]

    def cells_for(m, proto, n):
        m = m.lower()
        if proto == "rounds":
            name = f"{m}_{n}"
        elif proto == "open":
            name = f"open_{m}_{n}"
        else:
            name = f"indep_{m}_{n}_iso"
        return [c for c in VALID if c == name or (proto == "open" and c.startswith(name + "_"))]

    fig, ax = canvas(f"The experiment: {len(VALID)} cells, {sum(summary(c)['training_runs'] for c in VALID)} five-minute training runs",
                     "Model × organisation × team size; every cell has a budget of 36 training runs")
    x0, y0, cw, ch = 22, 12, 15, 13
    colors = {"rounds": ROUNDS_C, "open": OPEN_C, "isolated": GREEN}
    for j, (proto, n) in enumerate(cols):
        ax.text(x0 + j * cw + cw / 2, y0 + 3 * ch + 3, f"{n} agent{'s' if n > 1 else ''}", ha="center",
                fontsize=12.5, color=INK2)
    for proto, js in [("rounds", (0, 2)), ("open", (3, 5)), ("isolated", (6, 6))]:
        xa, xb = x0 + js[0] * cw + 1, x0 + (js[1] + 1) * cw - 1
        ax.plot([xa, xb], [y0 + 3 * ch + 7.5] * 2, color=colors[proto], lw=3)
        ax.text((xa + xb) / 2, y0 + 3 * ch + 9, proto, ha="center", fontsize=14, weight="bold", color=INK)
    for i, m in enumerate(models):
        y = y0 + (2 - i) * ch
        ax.text(x0 - 3, y + ch / 2, m, ha="right", va="center", fontsize=15, weight="bold", color=INK)
        for j, (proto, n) in enumerate(cols):
            got = cells_for(m, proto, n)
            x = x0 + j * cw
            if got:
                gains = [gain(c) for c in got]
                label = f"{gains[0]:.3f}" if len(got) == 1 else f"{min(gains):.3f}–{max(gains):.3f}\n({len(got)} repeats)"
                box(ax, x + 0.6, y + 0.6, cw - 1.2, ch - 1.2, label, fc=colors[proto], color="white", size=12)
            else:
                box(ax, x + 0.6, y + 0.6, cw - 1.2, ch - 1.2, "not run", fc=PALE, color=MUTED, size=11)
    fig.text(0.03, 0.03, "Number in each cell = gain in val_bpb (baseline − final). ~$285 of H100 time.",
             fontsize=12, color=MUTED)
    save(fig, "s3_setup_matrix.png")


# 6. which ideas paid off -----------------------------------------------------
CATS = [  # first match wins, on the primary change (text after "on ... base:" prefixes)
    ("attention window", ["window", "sssl", "all-s"]),
    ("heads / attention", ["head_dim", "heads", "gqa", "attention", "rope", "qk"]),
    ("value embeddings", ["value emb", "value-emb", " ve ", "ve ", "has_ve"]),
    ("MLP", ["mlp"]),
    ("depth / width", ["depth", "aspect", "width", "model_dim", "dim ", "layers", "u-net", "skip"]),
    ("batch size", ["batch"]),
    ("learning rate / schedule", ["lr", "learning rate", "warmdown", "warmup", "schedule", "decay", "cooldown"]),
    ("optimizer / numerics", ["muon", "adam", "beta", "weight decay", "lambda", "softcap", "init", "norm", "logit",
                              "precision", "bf16", "fp32", "softmax"]),
    ("throughput / systems", ["compile", "autotune", "dataloader", "prefetch", "flex", "kernel", "throughput"]),
]


def classify(desc):
    d = desc.lower()
    if ":" in d and "base" in d.split(":")[0]:
        d = d.split(":", 1)[1]
    d = " " + d + " "
    for name, keys in CATS:
        if any(k in d for k in keys):
            return name
    return "other"


def fig_ideas():
    tries, records = {}, {}
    crashes = 0
    for cell in VALID:
        best = baseline(cell)
        with open(RES / cell / "results.tsv") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["status"] != "ok" or not r["description"].strip():
                    crashes += r["status"] != "ok"
                    continue
                cat = classify(r["description"])
                tries[cat] = tries.get(cat, 0) + 1
                v = float(r["val_bpb"])
                if v < best - 0.0008:  # a new cell record beyond run noise
                    records[cat] = records.get(cat, 0) + 1
                    best = v
    cats = sorted(tries, key=lambda c: (records.get(c, 0), tries[c]))
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=0.22, right=0.9, top=0.8, bottom=0.15)
    title(fig, "Where the progress came from",
          "Every completed run in 23 cells, by the kind of change the agent made. Blue = set a new record for its cell.")
    ys = range(len(cats))
    rec = [records.get(c, 0) for c in cats]
    rest = [tries[c] - records.get(c, 0) for c in cats]
    ax.barh(ys, rec, height=0.62, color=BLUE, zorder=2)
    ax.barh(ys, rest, left=rec, height=0.62, color="#d9d8d2", zorder=2, edgecolor=SURF, linewidth=2)
    for y, c, r_ in zip(ys, cats, rec):
        ax.text(tries[c] + 2, y, f"{r_} of {tries[c]} set a record  ({100 * r_ / tries[c]:.0f}%)",
                va="center", fontsize=11.5, color=INK2)
    ax.set_yticks(list(ys), cats, fontsize=13)
    ax.set_xlabel("completed training runs")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(tries.values()) * 1.35)
    fig.text(0.22, 0.03, f"Classified by keywords in each run's one-line description. Record = beats the cell's best "
             f"by more than run noise (0.0008). {crashes} crashed runs left out.",
             fontsize=10.5, color=MUTED)
    save(fig, "fig6_ideas.png")


# 7. which effects are bigger than the noise ---------------------------------
def fig_effects():
    fin = lambda c: summary(c)["final_val_bpb"]
    effects = [  # (label, lower-is-better difference: a - b where positive means "first beats second")
        ("Opus vs Haiku, isolated 6", fin("indep_haiku_6_iso") - fin("indep_opus_6_iso")),
        ("Opus vs Sonnet, isolated 6", fin("indep_sonnet_6_iso") - fin("indep_opus_6_iso")),
        ("open vs rounds, Opus 3 agents", fin("opus_3") - fin("open_opus_3")),
        ("open vs rounds, Opus 1 agent", fin("opus_1") - fin("open_opus_1")),
        ("6 vs 1 agents, Opus rounds", fin("opus_1") - fin("opus_6")),
        ("6 vs 1 agents, Opus open", fin("open_opus_1") - (fin("open_opus_6") + fin("open_opus_6_s2")) / 2),
        ("open vs rounds, Opus 6 agents", fin("opus_6") - fin("open_opus_6")),
        ("sharing vs isolated, Opus 6", fin("indep_opus_6_iso") - fin("open_opus_6_iso")),
    ]
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.8, bottom=0.14)
    title(fig, "Which effects clear the noise",
          "Difference in final val_bpb between two arms (positive = the first arm did better)")
    ys = list(range(len(effects)))[::-1]
    ax.axvspan(-0.006, 0.006, color=PALE, zorder=0)
    ax.axvline(0, color=MUTED, lw=1)
    for y, (label, d) in zip(ys, effects):
        color = BLUE if abs(d) > 0.006 else MUTED
        ax.plot([0, d], [y, y], color=color, lw=2.5, zorder=2)
        ax.plot([d], [y], marker="o", markersize=10, color=color, markeredgecolor=SURF, markeredgewidth=2, zorder=3)
        ax.text(d + (0.0006 if d >= 0 else -0.0006), y + 0.28, f"{d:+.4f}", ha="left" if d >= 0 else "right",
                fontsize=11, color=INK2)
    ax.set_yticks(ys, [e[0] for e in effects], fontsize=12.5)
    ax.set_xlim(-0.006, 0.02)
    ax.set_xlabel("difference in val_bpb")
    ax.grid(axis="y", visible=False)
    ax.text(0.0058, ys[0] + 0.55, "grey band: too small to detect\nwith one seed per arm (±0.006)", ha="right",
            va="bottom", fontsize=11, color=INK2)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 1.4)
    fig.text(0.27, 0.03, "One seed per arm except Opus open 6 (two, averaged). Seed-to-seed spread of a whole cell: "
             "0.0015–0.0024.", fontsize=10.5, color=MUTED)
    save(fig, "fig7_effects.png")


if __name__ == "__main__":
    fig_setup_loop()
    fig_setup_protocols()
    fig_setup_matrix()
    fig_models()
    fig_headcount()
    fig_workflow()
    fig_sharing()
    fig_herding()
    fig_ideas()
    fig_effects()
