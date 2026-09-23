# batch-size & step-count economics

Explore the tokens-per-step vs number-of-optimizer-steps tradeoff under a fixed 5 min: shrink TOTAL_BATCH_SIZE (2^19 -> 2^18/2^17) with matched LR rescaling, and try shorter TRAINING sequence length (1024) while eval stays at 2048. I do NOT assume the given 2^19 batch or seq 2048 training is near-optimal, and I will not touch the optimizer family or depth/width as my primary lever.
