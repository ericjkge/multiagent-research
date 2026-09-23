# compute allocation: model shape vs optimizer steps

I will treat the 8-layer / 768-dim / 4x-MLP shape as an unjustified prior and sweep where the FLOPs go (width, depth, MLP ratio, GQA, window pattern) so that more optimizer steps fit in the fixed 300s. I will not assume the current shape is near compute-optimal, and I will keep TOTAL_BATCH_SIZE=2**17 fixed as a control rather than re-tuning batch size.
