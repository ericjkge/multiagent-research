# sequence-length curriculum & attention-cost scheduling

Train with a short context early (512/1024) and ramp to 2048 near the end, keeping tokens-per-step fixed by reshaping batches; attention is ~25% of FLOPs here so this buys extra optimizer steps for free. I do NOT assume a constant 2048 training context is optimal, and I will leave batch size and model shape to peers except where the ramp forces a change.
