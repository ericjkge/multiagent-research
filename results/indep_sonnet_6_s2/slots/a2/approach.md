# hparam/schedule tuning

Tune LR values, batch size, weight decay, and warmdown schedule within the fixed architecture; will not change depth/width/attention pattern initially. Will not assume baseline schedule (WARMDOWN_RATIO=0.5, flat LR before that) is optimal for a 5-min budget.
