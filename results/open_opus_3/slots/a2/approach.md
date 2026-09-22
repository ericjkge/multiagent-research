# attention & positional structure

Explore the attention block and positional/sequence structure: per-head attention output gating, value-embedding placement, rotary base, sliding-window pattern/width, GQA, and document-aware (varlen) masking so packed documents do not attend across boundaries. I will not assume the SSSL window pattern, base=10000 rope, alternating value embeddings, or cross-document attention are optimal; I will leave model shape (depth/width/batch) and the optimizer algorithm to peers.
