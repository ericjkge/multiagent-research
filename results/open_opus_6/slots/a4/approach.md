# model shape & architecture scaling

Withdrawing from batch-size (collision with a0/a2/a3). Instead: shape of the net under a 300s budget -- depth vs width (ASPECT_RATIO/DEPTH), MLP expansion ratio, head_dim/GQA kv-head sharing, and window pattern, judged on val_bpb not on params. I do NOT assume depth 8 / dim 512 / 4x MLP / MHA is near the compute-optimal point for 5 minutes, and I keep TOTAL_BATCH_SIZE at 2^19 so my results stay orthogonal to theirs.
