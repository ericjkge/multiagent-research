# compute-allocation co-design

Treat the 5-min budget as a FLOP/bandwidth allocation problem: measure where the step time actually goes (output head vs body vs attention), then re-balance model size, batch size, head/MLP shapes and the loss computation to buy more effective tokens. I will not assume the shipped DEPTH=8 / dim=512 / 2^19-token batch is optimal, nor that the f32 softcapped logit head is free.
