# batch-size & step-count economics

Explore the tokens-per-step vs number-of-optimizer-steps tradeoff under a 5-min budget: shrink TOTAL_BATCH_SIZE and/or ramp it during training, with matching LR/schedule adjustments. I do NOT assume the inherited 2**19 batch or the 50% warmdown are near-optimal, and I will not touch model size/architecture as the primary lever.
