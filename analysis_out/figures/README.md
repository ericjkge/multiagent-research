# Slide figures (Sep 24)

Regenerate with `python -m analysis.slides_figures`. Every number is read from `results/`.
1920 x 1080 PNGs, light background.

| file | what it shows | the line to say | caveat to say |
|---|---|---|---|
| `fig1_model_race.png` | Opus, Sonnet and Haiku, six isolated agents each, same machine, 36 runs | The model is the biggest lever: Opus 3.0% better, Sonnet 1.6%, Haiku 1.2%, and the weaker models crash a fifth of their runs | One seed per model |
| `fig2_headcount.png` | Progress of 3 and 6 agents relative to 1 agent, per model and workflow, within one machine | Six beat one for Opus (1.2x, 1.5x) and Sonnet rounds (3.1x); three agents do not reliably beat one; Haiku gets nothing | Extra agents add thinking, not compute: same 36 runs at every size. Haiku ratios are mostly noise |
| `fig3_herding.png` | What each of six Opus agents tried, blind rounds vs the shared log | Rounds herd (vertical stripes: all 12 proposals in rounds 2 and 3 included the same change); the shared log divides the work (horizontal stripes) | Idea families are hand-coded from the agents' run titles |
| `fig4_sharing_vs_isolation.png` | Six Opus agents sharing a log vs six sealed off from each other, same machine | Sharing everything did not beat seeing nothing: 2.76% vs 2.98% | One seed per arm; the 0.22 gap sits between the two seed-to-seed swings we measured (0.15, 0.24) |
| `fig5_relay.png` | How the six shared-log agents built their best result, record by record | One result, five authors: every record was adopted by 3 to 5 other agents | Collaboration happened; it did not beat isolation (fig 4) |
