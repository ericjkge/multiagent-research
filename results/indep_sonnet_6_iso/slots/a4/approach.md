# compute-optimal depth/width under fixed 5-min wall-clock

I will sweep DEPTH (and the derived model_dim/n_head via ASPECT_RATIO/HEAD_DIM) to find the depth that maximizes tokens-processed-quality tradeoff within the fixed time budget, plus complementary LR/schedule tuning. I will not assume the current DEPTH=8 default is optimal for this fixed wall-clock budget, since deeper models get fewer steps but may be more sample-efficient.
