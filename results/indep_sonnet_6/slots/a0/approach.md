# compute-optimal depth/width scan

Will scan DEPTH (and derived width via ASPECT_RATIO) to find the best model size for the fixed 5-min wall-clock budget, plus tune batch size/LR jointly with size. Will not assume the baseline DEPTH=8 is near-optimal for a time-budget (vs token-budget) regime.
