# compute allocation & shape at fixed FLOPs/token

Hold FLOPs-per-token (and hence tokens seen in 300s) roughly constant while re-allocating it: depth vs MLP expansion ratio vs attention window/KV-head budget, plus reinvesting savings into FLOP-free capacity (value embeddings). I will NOT assume depth=8 / dim=512 / MLP=4x / half-context windows is the optimal point, and I will not touch the optimizer or LR schedule.
