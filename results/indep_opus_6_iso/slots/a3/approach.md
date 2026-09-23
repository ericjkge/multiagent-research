# compute allocation: batch size x model shape

Re-tune the token/step tradeoff: how many optimizer steps the 5 minutes buys (TOTAL_BATCH_SIZE) and how to reinvest saved compute into width/depth and cheaper value-embedding budget. I do not assume the baseline's 524K-token batch, depth-8/dim-512 shape, or 4 full-vocab value embeddings are optimal.
