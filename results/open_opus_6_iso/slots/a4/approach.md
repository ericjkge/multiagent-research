# batch-size / step-count budget

Explore the token-batch vs optimizer-step tradeoff: TOTAL_BATCH_SIZE (currently 2^19 = 524K tokens/step, likely far above critical batch size for a ~30M-param model) plus the LR rescaling it implies and the warmup/warmdown shape. I will NOT assume the current depth/width, LR values or architecture are optimal, but I will hold architecture fixed and change only batch/step-schedule knobs so the effect is attributable.
