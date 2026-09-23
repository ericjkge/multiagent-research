# optimizer & LR/WD schedule internals

Tune the Muon+AdamW hyperparameters and the time-based schedules: per-group LR magnitudes (matrix/embedding/unembedding/scalar), warmup+warmdown shape and final-LR floor, Muon momentum/ns_steps, and the linearly-decaying weight decay. I do NOT assume the inherited LR set or the 50/50 trapezoid schedule is optimal; I keep batch size at 2^19 and model shape at depth 8 / dim 512 so my results stay orthogonal to the batch-size and shape agents.
