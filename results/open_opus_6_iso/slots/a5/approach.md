# residual-stream topology (U-net skips)

Add learned U-net style skip connections from early-layer outputs into the matching late layers (nanogpt-speedrun style encoder/decoder pairing), plus variations on the existing x0/resid lambda mixing and per-layer output gating. I will NOT touch batch size or the LR schedule shape (a4's slot), and will not assume depth 8 / SSSL windows are right for this topology.
