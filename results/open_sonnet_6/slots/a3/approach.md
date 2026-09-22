# optimization schedule & LR/WD hyperparameters

Keep DEPTH/ASPECT_RATIO/HEAD_DIM at baseline (leaving architecture size to a2/a4/a5) and instead tune LR magnitudes, warmup/warmdown schedule shape, weight-decay schedule, Adam betas, TOTAL_BATCH_SIZE and DEVICE_BATCH_SIZE. Will not assume the current linear warmdown or dmodel_lr_scale=1/sqrt is already optimal at fixed depth.
