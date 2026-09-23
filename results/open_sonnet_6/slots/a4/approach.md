# architecture micro-changes

At fixed DEPTH=8/ASPECT_RATIO/HEAD_DIM, will vary WINDOW_PATTERN (sliding window layout), MLP hidden expansion ratio, GQA n_kv_head<n_head, and logit softcap value. Will not touch model size/depth or LR/optimizer schedule (other agents' slots).
