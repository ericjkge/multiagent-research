# optimization scale (batch size x step count x LR schedule)

I will treat the token budget as fixed and re-allocate it: shrink TOTAL_BATCH_SIZE to buy more optimizer steps, then retune Muon/Adam LR and the warmdown shape to match. I will NOT assume the baseline 2**19-token batch or DEPTH=8 are near-optimal, and I will not chase architecture novelties (SwiGLU/MoE/new attention variants).
