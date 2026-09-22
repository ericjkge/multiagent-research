# architecture efficiency (attention/MLP shape)

Explore compute-per-token efficiency at roughly fixed optimizer settings: head count / head_dim, GQA (n_kv_head<n_head), MLP expansion ratio, sliding-window pattern, and value-embedding placement. I will NOT assume the current 4x MLP, 4 heads of dim 128, SSSL windows or full MHA are optimal, and I will not spend runs on pure LR/schedule sweeps (leaving that family to peers).
