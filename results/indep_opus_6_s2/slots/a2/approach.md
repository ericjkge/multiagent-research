# batch-size / step-count allocation

Explore how total tokens per optimizer step (and hence number of steps in the fixed 5 min) trades against per-step quality, plus the LR rescaling that goes with it; also cheap throughput changes (GQA, grad-accum=1) that buy more steps. I will not assume the baseline 2^19-token batch or the current depth-8/dim-512 shape are near-optimal for a 300 s budget.
