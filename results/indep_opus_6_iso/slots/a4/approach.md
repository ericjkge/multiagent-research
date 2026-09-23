# compute allocation: batch-size/step-count economics and model shape (depth, width, MLP ratio)

I will tune where the fixed 5-minute FLOP budget goes: number of optimizer steps (global batch size) vs model size/shape (depth, aspect ratio, MLP expansion, head_dim). I will NOT assume the baseline's depth-8/dim-512/4x-MLP/524k-token-batch is optimal, and I will not modify optimizer internals or the attention kernel.
