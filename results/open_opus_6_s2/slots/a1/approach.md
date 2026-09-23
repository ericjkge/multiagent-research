# model shape & capacity scaling

Hold batch size fixed at 2^19 (peers own that lever) and instead search the FLOP-optimal shape for a 300s run: DEPTH/ASPECT_RATIO/HEAD_DIM (e.g. depth 8->10/12, dim 512->640/768), GQA (n_kv_head < n_head) to buy width cheaply, and MLP expansion ratio. I do NOT assume depth-8/dim-512/head_dim-128/MHA is near-optimal, and I will not tune batch size or the LR schedule shape as my primary lever.
