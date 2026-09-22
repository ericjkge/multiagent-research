# optimizer & LR-schedule surface (final claim for a5)

Five of six agents piled into batch-economics and model-shape; I abandon both to keep coverage. My family: the optimizer/schedule surface at fixed batch 2^19 and fixed depth-8/dim-512 shape -- warmdown shape and FINAL_LR_FRAC, Muon MATRIX_LR and momentum schedule, EMBEDDING/UNEMBEDDING_LR balance, weight-decay schedule, and ns_steps. I do NOT assume the shipped LRs (matrix 0.04 / embed 0.6 / unembed 0.004) or the linear 50%-warmdown-to-zero are near-optimal, and I will not touch batch size or model shape (peers own those).
