# batch-size / update-count economics

Explore the number of optimizer updates per 5 minutes: shrink tokens-per-step (TOTAL_BATCH_SIZE, grad-accum) and co-tune LR/schedule so the same wall clock buys many more updates. I will not assume the baseline 524K-token batch or its LRs are near-optimal, and I will keep the architecture fixed while testing this axis.
