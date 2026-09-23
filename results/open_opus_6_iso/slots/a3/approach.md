# attention masking & data boundaries

The dataloader packs several documents per 2048-token row (each starting with BOS) but attention is fully causal across document boundaries, so every row mixes unrelated documents. I will add block-diagonal document masking via FA3 varlen (cu_seqlens derived from BOS positions), plus training-loop/step-efficiency fixes (data prefetch overlap, per-step syncs). I will NOT change depth/width, TOTAL_BATCH_SIZE, residual topology or the SSSL window pattern, so the effect composes with a0/a1/a4/a5/a2.
