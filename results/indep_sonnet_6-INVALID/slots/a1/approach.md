# model sizing / depth-width tradeoff

Will tune DEPTH, ASPECT_RATIO, HEAD_DIM, and DEVICE_BATCH_SIZE to find compute-optimal model size for the fixed 5-min budget, keeping optimizer/schedule fixed. Will not assume baseline DEPTH=8 is optimal; will not touch LR schedule shape unless model-size sweep is exhausted.
