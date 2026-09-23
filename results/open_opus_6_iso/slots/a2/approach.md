# attention shape & KV/value-embedding cost

Explore the attention/KV structure: GQA (n_kv_head < n_head, which also shrinks the huge per-layer value-embedding tables), head_dim (128 vs 64), and the sliding-window pattern/short-window length. I will NOT assume n_kv_head==n_head, head_dim=128 or SSSL/half-context windows are optimal, and I will hold batch size (a4) and residual topology (a5) fixed.
