# Results so far (2026-09-22 14:24)

- indep_haiku_6: OK — all invariants hold for results/indep_haiku_6
- indep_opus_6: OK — all invariants hold for results/indep_opus_6
- indep_opus_6_s2: OK — all invariants hold for results/indep_opus_6_s2
- indep_sonnet_6: OK — all invariants hold for results/indep_sonnet_6
- indep_sonnet_6_s2: OK — all invariants hold for results/indep_sonnet_6_s2
- open_opus_1: OK — all invariants hold for results/open_opus_1
- open_opus_3: OK — all invariants hold for results/open_opus_3
- open_opus_6: OK — all invariants hold for results/open_opus_6
- open_opus_6_s2:   note: open protocol: a5 claimed 7 runs, share was 6: one slot moved between agents when the budget was rebuilt after a resume; cell total unchanged
- opus_1: OK — all invariants hold for results/opus_1
- opus_3: OK — all invariants hold for results/opus_3
- opus_6: OK — all invariants hold for results/opus_6

```
cell           agents  BoN   baseline      final      gain  runs  rounds  crash%  simil.  agent $
-------------------------------------------------------------------------------------------------
indep_haiku_6       6    1   0.997359   0.976806  0.020553    36       1    30.6   0.146    23.33
indep_opus_6        6    1   0.997333   0.977649  0.019684    36       1     2.8   0.163    37.20
indep_opus_6_s2      6    1   0.997333   0.980071  0.017262    36       1     0.0   0.234    32.99
indep_sonnet_6      6    1   0.997333   0.988312  0.009021    36       1    11.1   0.293  2143.84
indep_sonnet_6_s2      6    1   0.997359   0.977046  0.020313    36       1    22.2   0.285  2077.04
open_opus_1         1    1   0.997359   0.980742  0.016617    36       1     0.0   0.000     5.48
open_opus_3         3    1   0.997359   0.982130  0.015229    36       1     2.8   0.199    18.99
open_opus_6         6    1   0.997333   0.977896  0.019437    36       1     0.0   0.183    35.93
open_opus_6_s2      6    1   0.997359   0.976408  0.020951    36       1     0.0   0.139    66.10
opus_1              1    3   0.997359   0.986277  0.011082    36       8     2.6   0.000    69.29
opus_3              3    2   0.997359   0.987792  0.009567    34       6     5.6   0.289   235.32
opus_6              6    1   0.997333   0.980496  0.016837    34       6     0.0   0.297   407.50
```

## Best val_bpb reached by run n (equal-compute comparison)

| cell | runs | best@29 | best@34 | best@36 | final |
|---|---:|---:|---:|---:|---:|
| indep_haiku_6 | 36 | 0.976806 | 0.976806 | 0.976806 | 0.976806 |
| indep_opus_6 | 36 | 0.981666 | 0.978436 | 0.977649 | 0.977649 |
| indep_opus_6_s2 | 36 | 0.981817 | 0.980071 | 0.980071 | 0.980071 |
| indep_sonnet_6 | 36 | 0.988312 | 0.988312 | 0.988312 | 0.988312 |
| indep_sonnet_6_s2 | 36 | 0.977046 | 0.977046 | 0.977046 | 0.977046 |
| open_opus_1 | 36 | 0.982289 | 0.981856 | 0.980742 | 0.980742 |
| open_opus_3 | 36 | 0.983331 | 0.982424 | 0.982130 | 0.98213 |
| open_opus_6 | 36 | 0.978654 | 0.977973 | 0.977896 | 0.977896 |
| open_opus_6_s2 | 36 | 0.977171 | 0.976408 | 0.976408 | 0.976408 |
| opus_1 | 36 | 0.986277 | 0.986277 | 0.986277 | 0.986277 |
| opus_3 | 34 | 0.987792 | 0.987792 | 0.987792 | 0.987792 |
| opus_6 | 34 | 0.981817 | 0.980496 | 0.980496 | 0.980496 |
