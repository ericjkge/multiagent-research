# output head, embeddings & loss parameterization

Orthogonal to the batch-size crowd (a0/a3/a4) and to a5's attention shape: I will work on the vocabulary-facing ends of the model and the loss — logit softcap value, lm_head scaling/init and its Adam LR, tied vs untied embeddings, value-embedding coverage (currently every other layer), and loss-side regularizers (z-loss / soft label smoothing). I will NOT change TOTAL_BATCH_SIZE, device batch, or attention shape as my primary lever, so my results compose with theirs.
