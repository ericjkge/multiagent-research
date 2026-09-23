# compute allocation: batch size x model shape

Treat the 5-minute budget as a compute-allocation problem: tune tokens-per-optimizer-step (TOTAL_BATCH_SIZE/grad-accum) and how the resulting FLOPs are split between depth, width and MLP ratio. I will NOT assume the inherited TOTAL_BATCH_SIZE=2**19 or DEPTH=8 are near-optimal for a 300s run, and I will not touch the optimizer algorithm itself (left to other families).
