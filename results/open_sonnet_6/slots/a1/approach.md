# model scale & batch size tuning

Will sweep DEPTH (model width/depth via ASPECT_RATIO) and TOTAL_BATCH_SIZE/DEVICE_BATCH_SIZE to find compute-optimal size for the fixed 5-min budget. Will not assume the baseline DEPTH=8 is optimal for this short budget; will not touch optimizer internals (Muon/AdamW math) or attention mechanism.
