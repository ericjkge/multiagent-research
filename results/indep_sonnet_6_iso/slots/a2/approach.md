# depth/width shape + LR schedule search

Will sweep DEPTH (and thus model width via ASPECT_RATIO formula) plus LR/warmdown/weight-decay schedule to find better compute-matched shape for the fixed 5-min budget. Will not assume the baseline DEPTH=8/ASPECT_RATIO=80 is compute-optimal, and will not touch optimizer internals (Muon/AdamW math) or attention mechanism itself.
