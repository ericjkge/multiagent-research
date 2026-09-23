# batch-size & step-count economics

Explore the tokens-per-step vs number-of-optimizer-steps tradeoff at fixed 5-min wall clock (TOTAL_BATCH_SIZE, grad-accum, device batch), and adapt LR/schedule to the resulting step count. I will NOT assume the inherited 2^19-token batch, 2 grad-accum steps, or that LRs tuned at that batch transfer.
