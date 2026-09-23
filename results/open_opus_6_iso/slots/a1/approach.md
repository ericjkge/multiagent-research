# model shape & capacity allocation

Sweep DEPTH / ASPECT_RATIO / HEAD_DIM / batch size to find the compute-optimal geometry for a 5-minute H100 budget (current: depth 8, dim 512, 4x128 heads). I will NOT assume the baseline geometry or the 2**19 token batch are optimal, and I will not touch optimizer internals beyond the dmodel LR rescale the code already applies.
