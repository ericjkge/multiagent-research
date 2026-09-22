# model shape / compute allocation

Sweep how the 5-minute FLOP budget is spent on shape: depth vs width (ASPECT_RATIO/DEPTH), head_dim/n_head, MLP expansion ratio, and total batch size / grad-accum. I will not assume the baseline 8L x 512d, 4x MLP, bs=2^19 point is near-optimal, and I will not touch the optimizer algorithm itself (only LR rescaling implied by shape).
