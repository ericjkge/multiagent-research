# compute-optimal model scaling

Treat the 5-min budget as a fixed FLOP budget and re-derive the width/depth/token tradeoff: at dim 768 x 8L the run sees only ~3-4 tokens/param, far under Chinchilla, so I will test narrower/shallower models that buy many more optimizer steps. I will not assume the current dim-768 depth-8 shape, HEAD_DIM 128, or the LR-scaling exponent are near-optimal.
