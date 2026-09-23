# residual topology & layer connectivity

Add modded-nanogpt-style U-net skip connections (learned gated skips from the first half of layers into the second half) on top of the existing x0 injection, plus explore per-layer residual scaling. I do NOT assume the current purely-sequential residual stream with only x0 re-injection is optimal; I will NOT touch batch size (three peers already hold that) or the optimizer family.
