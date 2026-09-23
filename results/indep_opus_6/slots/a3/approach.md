# batch-size / step-count allocation

Explore the tokens-per-step vs number-of-steps tradeoff under the fixed 5-min budget: TOTAL_BATCH_SIZE (currently 2^19, grad_accum=2) down to 2^18/2^17, with LR re-tuning as needed. I will NOT assume the current batch is near the critical batch size, and I will not change model size/depth as my primary lever.
