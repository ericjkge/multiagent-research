# depth/width scaling

Will search DEPTH and ASPECT_RATIO (model width) to find the best compute-optimal shape for the fixed 5-minute budget, holding the existing optimizer/architecture (Muon+AdamW, value embeddings, sliding window) fixed. Will not assume the default DEPTH=8 is optimal for this time budget; will not touch prepare.py or eval.
