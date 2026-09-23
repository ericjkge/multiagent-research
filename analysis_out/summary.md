# Results so far (2026-09-23 13:47)

**Read the correction at the top of case_studies.md first: the independent (indep_*) cells were not isolated (agents read the shared score table and peers' commits), so they do not measure the absence of communication.**

- indep_haiku_6: OK — all invariants hold
- indep_haiku_6_iso: OK — all invariants hold
- indep_haiku_6_s2: OK — all invariants hold
- indep_opus_6: OK — all invariants hold
- indep_opus_6_iso: OK — all invariants hold
- indep_opus_6_s2: OK — all invariants hold
- indep_sonnet_6: OK — all invariants hold
- indep_sonnet_6_iso: OK — all invariants hold
- indep_sonnet_6_s2: OK — all invariants hold
- open_opus_1: OK — all invariants hold
- open_opus_3: OK — all invariants hold
- open_opus_6: OK — all invariants hold
- open_opus_6_iso: OK — all invariants hold
- open_opus_6_s2: OK — all invariants hold (1 noted deviation(s), see verify)
- open_sonnet_1: OK — all invariants hold
- open_sonnet_3: OK — all invariants hold
- open_sonnet_6: OK — all invariants hold
- opus_1: OK — all invariants hold (1 noted deviation(s), see verify)
- opus_3: OK — all invariants hold (1 noted deviation(s), see verify)
- opus_6: OK — all invariants hold (2 noted deviation(s), see verify)
- sonnet_1: OK — all invariants hold
- sonnet_3: OK — all invariants hold
- sonnet_6: OK — all invariants hold (1 noted deviation(s), see verify)

```
cell           agents  BoN   baseline      final      gain  runs  rounds  crash%  simil.  agent $
-------------------------------------------------------------------------------------------------
indep_haiku_6       6    1   0.997359   0.976806  0.020553    36       1    30.6   0.146    23.33
indep_haiku_6_iso      6    1   1.012347   0.999826  0.012521    36       1    22.2   0.171     1.79
indep_haiku_6_s2      6    1   0.997333   0.995414  0.001919    36       1     0.0   0.152    16.33
indep_opus_6        6    1   0.997333   0.977649  0.019684    36       1     2.8   0.163    37.20
indep_opus_6_iso      6    1   1.012347   0.982146  0.030201    36       1     0.0   0.240    17.11
indep_opus_6_s2      6    1   0.997333   0.980071  0.017262    36       1     0.0   0.234    32.99
indep_sonnet_6      6    1   0.997333   0.988312  0.009021    36       1    11.1   0.293  2143.84
indep_sonnet_6_iso      6    1   1.012347   0.996232  0.016115    36       1    19.4   0.307     4.38
indep_sonnet_6_s2      6    1   0.997359   0.977046  0.020313    36       1    22.2   0.285  2077.04
open_opus_1         1    1   0.997359   0.980742  0.016617    36       1     0.0   0.000     5.48
open_opus_3         3    1   0.997359   0.982130  0.015229    36       1     2.8   0.199    18.99
open_opus_6         6    1   0.997333   0.977896  0.019437    36       1     0.0   0.183    35.93
open_opus_6_iso      6    1   1.012347   0.984417  0.027930    36       1     2.8   0.139    21.21
open_opus_6_s2      6    1   0.997359   0.976408  0.020951    36       1     0.0   0.139    66.10
open_sonnet_1       1    1   1.012531   0.979324  0.033207    36       1     2.8   0.000    12.30
open_sonnet_3       3    1   1.012531   0.993862  0.018669    36       1     0.0   0.253    48.71
open_sonnet_6       6    1   0.997333   0.985651  0.011682    36       1     8.3   0.207  1223.50
opus_1              1    3   0.997359   0.986277  0.011082    36       8     2.6   0.000    69.29
opus_3              3    2   0.997359   0.987792  0.009567    34       6     5.6   0.289   235.32
opus_6              6    1   0.997333   0.980496  0.016837    34       6     0.0   0.297   407.50
sonnet_1            1    5   1.014027   1.002279  0.011748    32       6     3.3   0.000    16.07
sonnet_3            3    2   1.014027   0.992950  0.021077    32       5     6.7   0.325    28.86
sonnet_6            6    1   1.014027   0.978041  0.035986    33       5     0.0   0.320    61.22
```

## Best val_bpb reached by run n (equal-compute comparison)

| cell | runs | best@29 | best@34 | best@36 | final |
|---|---:|---:|---:|---:|---:|
| indep_haiku_6 | 36 | 0.976806 | 0.976806 | 0.976806 | 0.976806 |
| indep_haiku_6_iso | 36 | 0.999826 | 0.999826 | 0.999826 | 0.999826 |
| indep_haiku_6_s2 | 36 | 0.995414 | 0.995414 | 0.995414 | 0.995414 |
| indep_opus_6 | 36 | 0.981666 | 0.978436 | 0.977649 | 0.977649 |
| indep_opus_6_iso | 36 | 0.982146 | 0.982146 | 0.982146 | 0.982146 |
| indep_opus_6_s2 | 36 | 0.981817 | 0.980071 | 0.980071 | 0.980071 |
| indep_sonnet_6 | 36 | 0.988312 | 0.988312 | 0.988312 | 0.988312 |
| indep_sonnet_6_iso | 36 | 0.998370 | 0.996232 | 0.996232 | 0.996232 |
| indep_sonnet_6_s2 | 36 | 0.977046 | 0.977046 | 0.977046 | 0.977046 |
| open_opus_1 | 36 | 0.982289 | 0.981856 | 0.980742 | 0.980742 |
| open_opus_3 | 36 | 0.983331 | 0.982424 | 0.982130 | 0.98213 |
| open_opus_6 | 36 | 0.978654 | 0.977973 | 0.977896 | 0.977896 |
| open_opus_6_iso | 36 | 0.986010 | 0.985510 | 0.984417 | 0.984417 |
| open_opus_6_s2 | 36 | 0.977171 | 0.976408 | 0.976408 | 0.976408 |
| open_sonnet_1 | 36 | 0.981159 | 0.979324 | 0.979324 | 0.979324 |
| open_sonnet_3 | 36 | 0.993862 | 0.993862 | 0.993862 | 0.993862 |
| open_sonnet_6 | 36 | 0.985651 | 0.985651 | 0.985651 | 0.985651 |
| opus_1 | 36 | 0.986277 | 0.986277 | 0.986277 | 0.986277 |
| opus_3 | 34 | 0.987792 | 0.987792 | 0.987792 | 0.987792 |
| opus_6 | 34 | 0.981817 | 0.980496 | 0.980496 | 0.980496 |
| sonnet_1 | 32 | 1.002279 | 1.002279 | 1.002279 | 1.002279 |
| sonnet_3 | 32 | 0.992950 | 0.992950 | 0.992950 | 0.99295 |
| sonnet_6 | 33 | 0.978041 | 0.978041 | 0.978041 | 0.978041 |
