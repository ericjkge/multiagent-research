# batch-size economics (tokens/step vs number of steps)

The baseline spends 524K tokens per optimizer step on a ~25M-matrix model; I will test whether halving/quartering TOTAL_BATCH_SIZE (more optimizer steps in the same 5 min) lowers val_bpb, with matched LR re-tuning, and then combine the best batch setting with one capacity change of my own. I do NOT assume the baseline batch is near the critical batch size, and I do NOT assume LRs transfer unchanged across batch sizes.
