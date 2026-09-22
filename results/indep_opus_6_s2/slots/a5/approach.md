# model shape & attention head geometry

Re-pick the width/depth/head_dim/MLP-ratio tradeoff under the fixed 5-min budget: head_dim 128->64 (more heads at dim 512), deeper-vs-wider aspect ratio, and MLP expansion ratio. I will NOT assume the baseline DEPTH=8 / HEAD_DIM=128 / 4x MLP point is compute-optimal for 300s, and I will not touch optimizer LRs unless a shape change demands it.
