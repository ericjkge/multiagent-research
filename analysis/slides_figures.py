"""Five slide figures for the Sep 24 presentation, built only from results/.

    python -m analysis.slides_figures        # writes analysis_out/figures/fig{1..5}_*.png

1. The race: Opus vs Sonnet vs Haiku, six isolated agents each, one machine.
2. Headcount: what 3 and 6 agents buy relative to 1, per model and workflow, within one machine.
3. Herding: what each of six Opus agents tried, blind rounds vs the shared log.
4. Sharing vs isolation: six Opus agents that share everything vs six that see nothing, one machine.
5. The relay: how six Opus agents on the shared log built one result.

Numbers come from each cell's summary.json and log.jsonl. The idea labels and families in
figure 3 are hand-coded from the agents' own run titles (listed below, auditable).
Palette: the dataviz reference instance (light); categorical sets validated with
validate_palette.js (models: all pairs; idea families: adjacent pairs, every cell also labelled).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "analysis_out" / "figures"

# ---- tokens (reference palette, light) ---------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID, AXIS, WASH = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
MODEL = {"Opus": "#2a78d6", "Sonnet": "#eb6834", "Haiku": "#1baf7a"}
FAMILY = {  # idea families for figure 3, slots 1-6 in order
    "batch": ("#2a78d6", "batch size"),
    "attn": ("#eb6834", "attention"),
    "shape": ("#1baf7a", "model shape"),
    "sched": ("#eda100", "learning rate & schedule"),
    "emb": ("#e87ba4", "embeddings & output head"),
    "speed": ("#008300", "speed & data"),
}
NOISE_BPB = 0.0008   # spread of six identical baseline runs (pod 1)
W, H, DPI = 12.8, 7.2, 150

_available = {f.name for f in fm.fontManager.ttflist}
FONT = next((f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in _available), "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "text.color": INK, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "savefig.facecolor": SURFACE})


# ---- data -------------------------------------------------------------------------------
def records(cell: str) -> list[dict]:
    return [json.loads(l) for l in open(RES / cell / "log.jsonl")]


def baseline(cell: str) -> float:
    return next(r for r in records(cell) if r.get("t") == "round_start")["baseline_bpb"]


def final(cell: str) -> float:
    return json.load(open(RES / cell / "summary.json"))["final_val_bpb"]


def gain_pct(cell: str) -> float:
    b = baseline(cell)
    return (b - final(cell)) / b * 100


def race(cell: str):
    """Open-protocol cell: runs in GPU order -> (x, best-so-far % better, crashes, per-run %)."""
    b = baseline(cell)
    cands = sorted((r for r in records(cell) if r.get("t") == "candidate"), key=lambda r: r["round"])
    assert [c["round"] for c in cands] == list(range(1, len(cands) + 1)), cell
    xs, ys, runs, best, crashes = [0], [0.0], [], b, 0
    for c in cands:
        ok = c.get("status") == "ok" and c.get("val_bpb") is not None
        crashes += not ok
        if ok:
            best = min(best, c["val_bpb"])
            runs.append((c["round"], (b - c["val_bpb"]) / b * 100, c))
        xs.append(c["round"])
        ys.append((b - best) / b * 100)
    return xs, ys, crashes, runs


# ---- chrome -----------------------------------------------------------------------------
def frame(title: str, subtitle: str, footnote: str | None = None):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    fig.text(0.055, 0.922, title, fontsize=25, fontweight="bold", color=INK, va="baseline")
    fig.text(0.055, 0.888, subtitle, fontsize=12.5, color=INK2, va="top", linespacing=1.45)
    if footnote:
        fig.text(0.055, 0.028, footnote, fontsize=9.5, color=MUTED, va="baseline", linespacing=1.4)
    return fig


def style(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="both", colors=MUTED, labelcolor=INK2, labelsize=11, length=0, pad=7)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)


def ylabel_top(ax, text):
    ax.text(0, 1.03, text, transform=ax.transAxes, fontsize=11, color=INK2, ha="left", va="bottom")


def endlabel(ax, x, y, head, sub, dx=0.35):
    ax.annotate(head, (x, y), xytext=(x + dx, y), fontsize=15, fontweight="bold", color=INK,
                va="bottom", annotation_clip=False)
    ax.annotate(sub, (x, y), xytext=(x + dx, y), fontsize=10.5, color=INK2, va="top",
                annotation_clip=False, linespacing=1.3)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / name)


# ---- 1. the race ------------------------------------------------------------------------
def fig1():
    cells = [("Opus", "indep_opus_6_iso", "winner: wider model and a 4x smaller batch"),
             ("Sonnet", "indep_sonnet_6_iso", "winner: wider model"),
             ("Haiku", "indep_haiku_6_iso", "winner: deeper model, embedding LR nudged")]
    b0 = baseline("indep_opus_6_iso")
    fig = frame("Opus runs away with the race",
                "Six agents of one model, each working alone, on the same GPU machine with the same 36 five-minute\n"
                "training runs. The line is the best result found so far.",
                f"Same machine for all three (untouched code scores {b0:.4f} bits per byte there). One search per model. Grey band: run-to-run noise "
                f"measured on another machine ({NOISE_BPB}).\nThis compares models working through this harness, crashes included; "
                "it is not a separate measure of idea quality. Transcript audit: no agent read another's scores, logs or code.")
    ax = fig.add_axes([0.075, 0.19, 0.60, 0.53])
    style(ax)
    noise = NOISE_BPB / b0 * 100
    ax.axhspan(-noise, noise, color=WASH, zorder=0)
    ax.text(35.6, -noise - 0.05, "noise", fontsize=9.5, color=MUTED, ha="right", va="top")
    ends = {}
    for model, cell, _ in cells:
        xs, ys, crashes, _ = race(cell)
        ax.step(xs, ys, where="post", color=MODEL[model], linewidth=2.8, solid_capstyle="round", zorder=3)
        ax.plot(xs[-1], ys[-1], "o", ms=9, color=MODEL[model], mec=SURFACE, mew=2, zorder=4)
        ends[model] = (ys[-1], crashes)
    offsets = {"Opus": 0.0, "Sonnet": 0.17, "Haiku": -0.17}   # nudge the two close labels apart
    for model, cell, what in cells:
        y, crashes = ends[model]
        crash_txt = "no crashed runs" if crashes == 0 else f"{crashes} of 36 runs crashed"
        endlabel(ax, 36, y + offsets[model], f"{model}  {y:.1f}% better", f"{what}\n{crash_txt}", dx=1.2)
    ax.set_xlim(0, 36.4)
    ax.set_ylim(-0.25, 3.4)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
    ax.set_xlabel("training runs used  (each run is 5 minutes on one H100)", fontsize=11, color=INK2, labelpad=8)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0%", "1%", "2%", "3%"])
    ylabel_top(ax, "better than the starting code  (fewer bits per byte)")
    save(fig, "fig1_model_race.png")


# ---- 2. headcount -----------------------------------------------------------------------
def fig2():
    fams = [  # (model, workflow, {agents: [cells]}, hollow)
        ("Opus", "shared log", {1: ["open_opus_1"], 3: ["open_opus_3"], 6: ["open_opus_6", "open_opus_6_s2"]}, False),
        ("Opus", "rounds", {1: ["opus_1"], 3: ["opus_3"], 6: ["opus_6"]}, False),
        ("Sonnet", "rounds", {1: ["sonnet_1"], 3: ["sonnet_3"], 6: ["sonnet_6"]}, False),
        ("Sonnet", "shared log", {1: ["open_sonnet_1"], 3: ["open_sonnet_3"]}, False),
        ("Haiku", "shared log", {1: ["open_haiku_1"], 3: ["open_haiku_3"], 6: ["open_haiku_6"]}, False),
        ("Haiku", "rounds", {1: ["haiku_1"], 3: ["haiku_3"], 6: ["haiku_6"]}, True),
    ]
    fig = frame("Adding agents is not a reliable multiplier",
                "Each line is one model and workflow on one machine, with the same 36 experiments at every team size, so extra agents\n"
                "add thinking, not compute. At three agents, three lines rose and three fell; at six, three rose and both Haiku lines fell.",
                "Progress = improvement over the untouched code, divided by the one-agent improvement on the same line. One search per point.\n"
                "Haiku's gains are only 2 to 7 times run noise, so its ratios swing on noise. Round cells ended short of 36 runs: Opus 34 (3 and 6 agents),\n"
                "Sonnet 32 to 33, Haiku 20 to 28 (hollow). Opus shared log at 6 agents is the mean of two seeds (1.17x and 1.26x).")
    ax = fig.add_axes([0.075, 0.19, 0.56, 0.53])
    style(ax)
    ax.axhline(1, color=INK2, linewidth=1.1, zorder=2)
    labels = []
    for model, flow, pts, hollow in fams:
        base = gain_pct(pts[1][0])
        xs, ys = [], []
        for n in sorted(pts):
            vals = [gain_pct(c) / base for c in pts[n]]
            xs.append(n)
            ys.append(sum(vals) / len(vals))
            if len(vals) > 1:
                ax.plot([n, n], [min(vals), max(vals)], color=MODEL[model], linewidth=1.4, zorder=3)
                ax.plot([n] * len(vals), vals, "o", ms=5, color=MODEL[model], mec=SURFACE, mew=1.2, zorder=4)
        marker = "s" if flow == "rounds" else "o"
        ax.plot(xs, ys, color=MODEL[model], linewidth=2.6, zorder=3)
        ax.plot(xs[1:], ys[1:], marker, ms=9, color=MODEL[model],
                mfc=SURFACE if hollow else MODEL[model], mec=MODEL[model] if hollow else SURFACE, mew=2, zorder=5)
        labels.append((xs[-1], ys[-1], f"{model}, {flow}", f"{ys[-1]:.1f}x"))
    ax.plot(1, 1, "o", ms=10, color=INK2, mec=SURFACE, mew=2, zorder=6)
    for x, y, name, val in labels:
        if x == 6:
            ax.annotate(f"{val}  ", (x, y), xytext=(6.25, y), fontsize=14, fontweight="bold", color=INK,
                        va="center", annotation_clip=False)
            ax.annotate(name, (x, y), xytext=(6.78, y), fontsize=11, color=INK2, va="center", annotation_clip=False)
        else:  # the Sonnet shared-log line ends at 3 agents
            ax.annotate(f"{val}  {name}\n(no 6-agent run on the same machine)", (x, y), xytext=(x - 0.1, y - 0.12),
                        fontsize=10.5, color=INK2, ha="right", va="top", linespacing=1.3)
    ax.set_xlim(0.7, 6.15)
    ax.set_ylim(0, 3.4)
    ax.set_xticks([1, 3, 6])
    ax.set_xticklabels(["1 agent", "3 agents", "6 agents"], fontsize=12)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0x", "1x", "2x", "3x"])
    ylabel_top(ax, "progress relative to one agent of the same model and workflow")
    save(fig, "fig2_headcount.png")


# ---- 3. herding -------------------------------------------------------------------------
# (agent) -> six (label, family, star, note) in time order. Labels paraphrase the agents' own titles.
ROUNDS_OPUS6 = {  # results/opus_6, proposals by round; star = round winner (became the new baseline)
    "a0": [("batch ½", "batch", 0, ""), ("head dim 64", "attn", 0, ""), ("head dim 64", "attn", 0, ""),
           ("warmdown 0.7", "sched", 1, ""), ("device batch\n128", "speed", 0, ""), ("value emb.\nevery layer", "emb", 0, "")],
    "a1": [("batch ½", "batch", 0, ""), ("depth 12\n+ head dim 64", "attn", 0, ""), ("head dim 64", "attn", 0, ""),
           ("warmdown 0.7", "sched", 0, ""), ("device batch\n128", "speed", 1, ""), ("autotune\ncompile", "speed", 1, "")],
    "a2": [("batch ½", "batch", 0, ""), ("head dim 64", "attn", 0, ""), ("head dim 64", "attn", 0, "not run"),
           ("warmdown 0.7", "sched", 0, ""), ("window 512", "attn", 0, ""), ("autotune\ncompile", "speed", 0, "not run")],
    "a3": [("U-net skips", "shape", 0, ""), ("head dim 64", "attn", 0, ""), ("head dim 64", "attn", 0, ""),
           ("warmdown 0.7", "sched", 0, ""), ("window 512", "attn", 0, ""), ("autotune\ncompile", "speed", 0, "")],
    "a4": [("batch ½", "batch", 0, ""), ("depth 12\n+ head dim 64", "attn", 0, ""), ("head dim 64", "attn", 0, ""),
           ("warmdown 0.7", "sched", 0, ""), ("window 512", "attn", 0, ""), ("shared\nvalue emb.", "emb", 0, "")],
    "a5": [("batch ½", "batch", 1, ""), ("head dim 64", "attn", 0, ""), ("narrower\n+ head dim 64", "shape", 0, ""),
           ("warmdown 0.7", "sched", 0, ""), ("RoPE base", "attn", 0, ""), ("bf16 logits", "emb", 0, "")],
}
OPEN_OPUS6 = {  # results/open_opus_6, each agent's 1st..6th run (GPU rotation); star = new best in the cell
    "a0": [("LR x1.25", "sched", 0, ""), ("warmdown 0.8", "sched", 0, ""), ("window\n+ warmdown", "sched", 0, ""),
           ("warmdown 1.0", "sched", 1, ""), ("peak LR x1.4", "sched", 0, ""), ("warmdown 1.0\n(width 768)", "sched", 1, "")],
    "a1": [("seq-length\ncurriculum", "speed", 0, ""), ("faster\ndataloader", "speed", 0, ""), ("stack\n+ fast loader", "speed", 0, ""),
           ("stack\n+ fast loader", "speed", 0, ""), ("weight EMA", "sched", 0, ""), ("width 896", "shape", 0, "")],
    "a2": [("value emb.\nevery layer", "emb", 0, ""), ("drop logit\nsoftcap", "emb", 0, ""), ("softcap 8", "emb", 0, ""),
           ("drop value\nemb.", "emb", 0, ""), ("value-emb.\ngate", "emb", 0, ""), ("value-emb.\nplacement", "emb", 1, "")],
    "a3": [("batch ½", "batch", 1, ""), ("batch ¼", "batch", 0, ""), ("batch ramp", "batch", 0, ""),
           ("batch ¼\nagain", "batch", 0, ""), ("tail weight\naverage", "sched", 0, ""), ("MLP 3x", "shape", 0, "")],
    "a4": [("depth 10", "shape", 0, ""), ("deeper\n+ narrower", "shape", 0, ""), ("shallow\n+ wide", "shape", 1, ""),
           ("depth 5,\nwidth 768", "shape", 0, ""), ("full stack", "shape", 0, ""), ("MLP 3x", "shape", 0, "")],
    "a5": [("head dim 64", "attn", 0, ""), ("window 256", "attn", 1, ""), ("window\npattern", "attn", 0, ""),
           ("window\n+ warmdown", "attn", 0, ""), ("width 768", "shape", 1, ""), ("width 896", "shape", 0, "")],
}


def _ink_on(hex_color: str) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#ffffff" if (1.05 / (lum + 0.05)) >= ((lum + 0.05) / 0.05) else INK


def _grid(ax, x0, grid, col_label):
    agents = ["a0", "a1", "a2", "a3", "a4", "a5"]
    for r, a in enumerate(agents):
        ax.text(x0 - 0.18, r + 0.5, a, fontsize=11, color=INK2, ha="right", va="center")
        for c, (label, fam, star, note) in enumerate(grid[a]):
            color = FAMILY[fam][0]
            alpha = 0.32 if note else 1.0
            ax.add_patch(FancyBboxPatch((x0 + c + 0.04, r + 0.05), 0.92, 0.90, boxstyle="round,pad=0,rounding_size=0.08",
                                        facecolor=color, edgecolor="none", alpha=alpha))
            txt = label + (f"\n({note})" if note else "")
            ax.text(x0 + c + 0.5, r + 0.53, txt, fontsize=8.0, ha="center", va="center", linespacing=1.15,
                    color=INK if note else _ink_on(color))
            if star:
                ax.plot(x0 + c + 0.9, r + 0.1, marker="*", ms=11, color=INK, mec=SURFACE, mew=1.3, zorder=5)
    for c in range(6):
        ax.text(x0 + c + 0.5, -0.12, f"{col_label} {c + 1}", fontsize=10, color=INK2, ha="center", va="bottom")


def _distinct(grid):
    return [len({grid[a][c][1] for a in grid}) for c in range(6)]


def fig3():
    d_r, d_o = _distinct(ROUNDS_OPUS6), _distinct(OPEN_OPUS6)
    fig = frame("Blind rounds herd. A shared log splits the work.",
                "Six Opus agents, 36 experiments per workflow. Each square is one experiment, colored by what it changed.\n"
                "Rounds: columns are one color, everyone tried the same thing. Shared log: rows are mostly one color, each agent owned a direction.",
                f"Directions tried per round: rounds {', '.join(map(str, d_r))}; shared log {', '.join(map(str, d_o))} "
                "(converging on width at the end). Labels paraphrase the agents' own run titles; families are hand-coded.\n"
                "Rounds protocol: proposals are blind, all six run, one winner becomes everyone's new baseline. "
                "'not run': the session ended before training (harness bug, never a winner).")
    ax = fig.add_axes([0.035, 0.2, 0.93, 0.58])
    ax.set_xlim(-0.55, 13.05)
    ax.set_ylim(6.35, -1.0)
    ax.axis("off")
    _grid(ax, 0, ROUNDS_OPUS6, "round")
    _grid(ax, 7.0, OPEN_OPUS6, "run")
    ax.text(0.04, -0.72, "Blind rounds", fontsize=15, fontweight="bold", color=INK, va="bottom")
    ax.text(7.04, -0.72, "Shared research log", fontsize=15, fontweight="bold", color=INK, va="bottom")
    ax.text(0.04, 6.12, "Rounds 2 and 3: all 12 proposals included the same head-dim change,\nand neither round found a winner.",
            fontsize=10, color=INK2, va="top", linespacing=1.35)
    ax.text(7.04, 6.12, "Before the first run, 4 agents claimed 'batch size', saw the collision\nin the log, and 3 switched to other directions.",
            fontsize=10, color=INK2, va="top", linespacing=1.35)
    # legend row
    rend = fig.canvas.get_renderer()
    x, yl = 0.055, 0.118
    for key, (color, name) in FAMILY.items():
        fig.patches.append(FancyBboxPatch((x, yl - 0.011), 0.012, 0.022, boxstyle="round,pad=0,rounding_size=0.003",
                                          transform=fig.transFigure, facecolor=color, edgecolor="none"))
        tx = fig.text(x + 0.017, yl, name, fontsize=10.5, color=INK2, va="center")
        x = tx.get_window_extent(renderer=rend).x1 / fig.bbox.width + 0.024
    fig.add_artist(Line2D([x + 0.006], [yl], marker="*", ms=12, color=INK, mec=SURFACE, mew=1.2, transform=fig.transFigure))
    fig.text(x + 0.017, yl, "became the new best", fontsize=10.5, color=INK2, va="center")
    save(fig, "fig3_herding.png")


# ---- 4. sharing vs isolation ------------------------------------------------------------
def fig4():
    b0 = baseline("open_opus_6_iso")
    fig = frame("Sharing everything did not beat seeing nothing",
                "Six Opus agents on one machine, 36 experiments each. One team shared a research log (scores, findings, code).\n"
                "The other ran in sealed-off copies with no way to see each other.",
                "One search per arm: the isolated team did not do worse, which is not the same as the two being equal, and no threshold follows from one pair.\n"
                "An earlier 'independent' control leaked (agents could read a shared score table). This rerun replaced it; a transcript audit found no peer access.")
    ax = fig.add_axes([0.075, 0.19, 0.56, 0.53])
    style(ax)
    rows = [("shared a research log", "open_opus_6_iso", MODEL["Opus"]),
            ("saw nothing of each other", "indep_opus_6_iso", INK2)]
    finals = {}
    for name, cell, color in rows:
        xs, ys, crashes, _ = race(cell)
        ax.step(xs, ys, where="post", color=color, linewidth=2.8, zorder=3)
        ax.plot(xs[-1], ys[-1], "o", ms=9, color=color, mec=SURFACE, mew=2, zorder=4)
        finals[name] = ys[-1]
    for name, _, _ in rows:
        y = finals[name]
        dy = 0.07 if y == max(finals.values()) else -0.07
        ax.annotate(f"{y:.2f}%", (36, y), xytext=(37.0, y + dy), fontsize=15, fontweight="bold",
                    va="center", annotation_clip=False)
        ax.annotate(name, (36, y), xytext=(41.2, y + dy), fontsize=11, color=INK2, va="center", annotation_clip=False)
    ax.set_xlim(0, 36.4)
    ax.set_ylim(-0.15, 3.4)
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0%", "1%", "2%", "3%"])
    ax.set_xlabel("training runs used  (each run is 5 minutes on one H100)", fontsize=11, color=INK2, labelpad=8)
    ylabel_top(ax, "better than the starting code  (fewer bits per byte)")
    # yardstick: the gap next to how far one setup moves between two seeds (Monday)
    gap = abs(finals["shared a research log"] - finals["saw nothing of each other"])
    seeds = sorted([gain_pct("open_opus_6"), gain_pct("open_opus_6_s2")])
    wobble_open = seeds[1] - seeds[0]
    wobble_leaky = abs(gain_pct("indep_opus_6") - gain_pct("indep_opus_6_s2"))
    ya = fig.add_axes([0.715, 0.26, 0.26, 0.24])
    ya.set_xlim(0, 1)
    ya.set_ylim(0, 0.34)
    ya.axis("off")
    ya.text(0, 0.40, "How big is that gap?", fontsize=12.5, fontweight="bold", color=INK, va="bottom")
    ya.text(0, 0.375, "in points; grey bars are from another machine", fontsize=9.5, color=MUTED, va="top")
    bars = [(gap, "this gap", MODEL["Opus"]),
            (wobble_open, "shared log,\nrerun with a\nnew seed", AXIS),
            (wobble_leaky, "earlier control,\nrerun with a\nnew seed", AXIS)]
    for i, (h, label, color) in enumerate(bars):
        x = 0.06 + i * 0.3
        ya.add_patch(FancyBboxPatch((x, 0), 0.14, h, boxstyle="round,pad=0,rounding_size=0.012",
                                    facecolor=color, edgecolor="none", mutation_aspect=0.3))
        ya.text(x + 0.07, h + 0.008, f"{h:.2f}", fontsize=11.5, fontweight="bold", color=INK, ha="center", va="bottom")
        ya.text(x + 0.07, -0.015, label, fontsize=9, color=INK2, ha="center", va="top", linespacing=1.25)
    save(fig, "fig4_sharing_vs_isolation.png")


# ---- 5. the relay -----------------------------------------------------------------------
def fig5():
    cell = "open_opus_6"
    recs = records(cell)
    b0 = baseline(cell)
    xs, ys, _, runs = race(cell)
    adopted = {}
    for r in recs:
        if r.get("t") == "adoption":
            adopted[r["commit"][:7]] = adopted.get(r["commit"][:7], 0) + 1
    steps, best = [], b0
    for x, pct, c in runs:
        if c["val_bpb"] < best:
            best = c["val_bpb"]
            steps.append((x, pct, c))
    story = {  # plain-English line per record, from the run titles
        1: ("a3", "halves the batch: twice the optimizer steps"),
        8: ("a5", "shortens the attention window, on a3's batch"),
        15: ("a4", "goes shallow and wide (6 layers, width 640)"),
        22: ("a0", "stretches the learning-rate cooldown"),
        26: ("a5", "widens again, to 768"),
        34: ("a0", "cooldown again, on the 768 model"),
        35: ("a2", "moves the value embeddings: final best"),
    }
    # runs 34 and 35 are back-to-back records 0.008 points apart: one numbered step, both authors named
    shown = [s for s in steps if s[0] != 34]
    authors = sorted({story[x][0] for x, _, _ in steps})
    fig = frame("One result, five authors",
                "Six Opus agents sharing a research log, 36 experiments. Each numbered step is a new best, and the next agents built on it.\n"
                f"The final model stacks changes found by {len(authors)} of the 6 agents.",
                f"Grey dots: every experiment ({len(runs) - len(steps)} of {len(runs)} did not set a new best). 'Adopted by': other agents "
                "pulled that code into their own work after reading its score in the log.\n"
                "Collaboration clearly happened. Whether it beats working alone is chart 4: in the one matched pair, the isolated team found the better result.")
    ax = fig.add_axes([0.075, 0.19, 0.50, 0.53])
    style(ax)
    for x, pct, c in runs:
        ax.plot(x, pct, "o", ms=6.5, color=AXIS, mec=SURFACE, mew=1, zorder=2)
    ax.step(xs, ys, where="post", color=MODEL["Opus"], linewidth=2.8, zorder=3)
    ax.plot(34, next(p for x, p, _ in steps if x == 34), "o", ms=9, color=MODEL["Opus"], mec=SURFACE, mew=1.5, zorder=4)
    for i, (x, pct, c) in enumerate(shown, start=1):
        ax.plot(x, pct, "o", ms=17, color=MODEL["Opus"], mec=SURFACE, mew=2, zorder=5)
        ax.text(x, pct, str(i), fontsize=9.5, fontweight="bold", color="#ffffff", ha="center", va="center", zorder=6)
    ax.set_xlim(0, 37)
    ax.set_ylim(-0.8, 2.4)
    ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
    ax.set_yticks([-0.5, 0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(["-0.5%", "0%", "0.5%", "1%", "1.5%", "2%"])
    ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
    ax.set_xlabel("experiment number  (about 6 minutes each, 3.4 hours in all)", fontsize=11, color=INK2, labelpad=8)
    ylabel_top(ax, "better than the starting code")
    # the story, as a numbered list to the right
    y = 0.705
    story[35] = ("a0 and a2", "stack the cooldown and move the value embeddings")
    for i, (x, pct, c) in enumerate(shown, start=1):
        who, what = story[x]
        n = adopted.get(c["commit"][:7], 0)
        fig.patches.append(FancyBboxPatch((0.612, y - 0.006), 0.019, 0.034, boxstyle="round,pad=0,rounding_size=0.009",
                                          transform=fig.transFigure, facecolor=MODEL["Opus"], edgecolor="none"))
        fig.text(0.6215, y + 0.011, str(i), fontsize=9.5, fontweight="bold", color="#ffffff", ha="center", va="center")
        fig.text(0.642, y + 0.011, f"{who} {what}", fontsize=11, color=INK, va="center")
        runs_txt = "runs 34 and 35" if x == 35 else f"run {x}"
        sub = f"{runs_txt},  {pct:.2f}% better" + (f",  adopted by {n} other agent{'s' if n != 1 else ''}" if n else
                                                   (",  the final best" if x == 35 else ""))
        fig.text(0.642, y - 0.024, sub, fontsize=9.5, color=INK2, va="center")
        y -= 0.09
    save(fig, "fig5_relay.png")


def main() -> int:
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
