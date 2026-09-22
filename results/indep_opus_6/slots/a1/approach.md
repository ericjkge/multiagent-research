# optimization scaling: batch size, step count, LR schedule shape

I will treat the baseline's 2^19-token batch (~950 steps in 5 min) and its 50% linear warmdown as untuned, and sweep the batch/step-count tradeoff plus schedule shape, retuning LR only as needed. I will not assume the architecture needs changing; architecture edits only as cheap free-flop tweaks (e.g. head_dim at fixed n_embd).
