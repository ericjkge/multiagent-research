# residual-stream topology

Add U-Net-style encoder->decoder skip connections with learned per-skip gates on top of the existing x0/resid lambdas, then tune the depth/skip interaction. I do NOT assume the current depth-8 / dim-512 / 4-head shape or the LR set are optimal, but I will change them only one at a time on top of whatever skip topology measures best.
