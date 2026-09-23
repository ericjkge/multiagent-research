# depth/width scaling for fixed time budget

I will sweep DEPTH (and derived width via ASPECT_RATIO/HEAD_DIM) to find the compute-optimal model size for a fixed 5-minute wall-clock budget, since the baseline depth=8 was not necessarily tuned for a time-limited (vs token-limited) regime. I will not assume LR/optimizer hyperparams are already optimal for the new sizes and will do a light LR re-check if depth changes a lot.
