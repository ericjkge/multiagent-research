# optimizer & schedule tuning

Tune optimizer/schedule hyperparameters at fixed DEPTH=8 architecture: WARMDOWN_RATIO, WEIGHT_DECAY, ADAM_BETAS, MATRIX_LR, muon momentum/ns_steps schedule, and TOTAL_BATCH_SIZE/DEVICE_BATCH_SIZE tradeoffs. Will not change model architecture (depth/width/window pattern) unless schedule tuning plateaus.
