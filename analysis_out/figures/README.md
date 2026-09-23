# Slide figures (Sep 24)

Regenerate with `python -m analysis.slides_figures`. Every number is read from `results/`.
1920 x 1080 PNGs, light background.

| file | what it shows | the line to say | caveat to say |
|---|---|---|---|
| `fig1_model_race.png` | Opus, Sonnet and Haiku, six isolated agents each, same machine, 36 runs | Same machine and budget: Opus 3.0% better, Sonnet 1.6%, Haiku 1.2%; Sonnet and Haiku lost about a fifth of their runs | One search per model; models through this harness, not idea quality alone |
| `fig2_headcount.png` | Progress of 3 and 6 agents relative to 1 agent, per model and workflow, within one machine | Mixed: six beat one for Opus (1.2x, 1.5x) and Sonnet rounds (3.1x); three beat one in two lines; Haiku shared log got worse at six | Extra agents add thinking, not compute: same 36 runs at every size. Haiku ratios are mostly noise |
| `fig3_herding.png` | What each of six Opus agents actually ran, rounds vs the shared log | Rounds move as a pack (columns: in round 2 four agents ran the same 14-layer model); the shared log divides the work (rows) | Rounds are not communication-free: agents see all proposals and messages before running, and proposals herded more than runs. Families hand-coded; one search per workflow |
| `fig4_sharing_vs_isolation.png` | Six Opus agents sharing a log vs six sealed off from each other, same machine | Sharing everything did not beat seeing nothing: 2.76% vs 2.98% | One search per arm: the isolated team did not do worse; that is not equivalence, and no threshold follows from one pair |
| `fig5_relay.png` | How the six shared-log agents built their best result, record by record | One result, five authors: every record was adopted by 3 to 5 other agents | Collaboration happened; it did not beat isolation (fig 4) |
