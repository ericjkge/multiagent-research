# depth/width scaling + LR schedule tuning

Will sweep DEPTH (model size) and optimizer LRs/warmdown under the fixed 5-min budget to find the best compute-optimal point; will not touch attention mechanism, windowing, or value-embedding architecture.
