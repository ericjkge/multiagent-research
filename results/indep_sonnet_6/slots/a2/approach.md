# depth/width scaling + batch size tradeoff

I will sweep model depth (and derived width) and device batch size / grad-accum under the fixed 5-min budget to find a better compute-optimal point than the DEPTH=8 baseline; I will not assume the current LR schedule is optimal for a different depth, so I'll rescale LR via the existing 1/sqrt(dmodel) rule already in the code.
