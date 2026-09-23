# compute allocation & throughput

Hold architecture and optimizer math fixed; buy more optimizer steps and more tokens inside the 300s: total/device batch size, grad-accum count, cheaper loss head (chunked/bf16 logits), MFU. I will NOT assume the baseline's 2^19-token batch or DEVICE_BATCH_SIZE=128 are near-optimal for a 5-minute budget.
