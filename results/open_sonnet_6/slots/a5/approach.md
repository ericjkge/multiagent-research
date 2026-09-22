# attention/MLP micro-architecture

At fixed DEPTH=8/ASPECT_RATIO=64 baseline shape, vary window_pattern (e.g. all-long vs SSSL vs SSSSSSL), GQA n_kv_head<n_head, HEAD_DIM, and MLP hidden expansion ratio. Will not touch optimizer LR/schedule math or overall depth/width.
