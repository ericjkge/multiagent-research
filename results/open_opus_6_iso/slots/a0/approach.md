# compute allocation (batch size x model shape x step count)

Explore how the 5-minute compute is split: tokens-per-step (TOTAL_BATCH_SIZE=2^19 gives only ~1k steps) vs number of optimizer steps, and depth/width aspect ratio. I will NOT assume the current batch size or the ASPECT_RATIO=64 shape is near-optimal, and will not touch optimizer internals or add architecture.
