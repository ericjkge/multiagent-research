# attention microarchitecture

Vary the attention shape and internals at fixed FLOPs: head_dim/head count (currently 128-dim x 4 heads at d=512), sliding-window pattern, and attention-level gating/sinks. I will NOT assume the tuned defaults (HEAD_DIM=128, SSSL, MHA) are optimal, and will not change depth/width or the optimizer except as a controlled follow-up.
