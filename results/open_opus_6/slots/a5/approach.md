# attention shape & block topology

Vary head_dim/head count (currently only 4 heads of 128 at d=512), GQA kv-heads, window pattern, and MLP expansion ratio / depth-width split. I will NOT touch the LR schedule, batch size, or optimizer constants (leaving those to peers) except where a shape change forces it.
