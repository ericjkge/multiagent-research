# compute allocation: batch size x model shape

Jointly retune the tokens-per-step / number-of-steps tradeoff (TOTAL_BATCH_SIZE) with model depth+width and the LRs that must co-scale with them. I will not assume the shipped TOTAL_BATCH_SIZE=2^19 or DEPTH=8 are jointly optimal for a 300s budget, and I will not touch the attention/optimizer internals unless the shape experiments plateau.
