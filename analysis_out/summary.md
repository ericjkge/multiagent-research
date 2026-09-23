# Results so far (2026-09-23 14:01)

**Interpretation and exclusions.** Read [ONE_PAGE.md](ONE_PAGE.md) and [case_studies.md](case_studies.md).
The six original `indep_{haiku,sonnet,opus}_6` and `_s2` searches cannot serve as independent controls:
the harness exposed peers' scores and code, with direct contamination documented in the Opus and Haiku
archives. The three `indep_*_6_iso` reruns are separate repaired controls with clean recorded isolation
scans; `open_opus_6_iso` intentionally permits sharing. There are 29 archived search summaries below,
not 29 equally valid causal comparisons. Verifier OK means its implemented checks passed, sometimes
with tolerated deviations; it does not override these exclusions.

The numerical tables below are retained from the existing report. Round-based crash percentages use
candidate records and can differ from failed GPU executions. Final scores are descriptive; actual
attempts, hardware, resumes and validation-selection effects limit comparisons.

- haiku_1: OK — all invariants hold (8 noted deviation(s), see verify)
- haiku_3: OK — all invariants hold (10 noted deviation(s), see verify)
- haiku_6: OK — all invariants hold (11 noted deviation(s), see verify)
- indep_haiku_6: OK — all invariants hold
- indep_haiku_6_iso: OK — all invariants hold
- indep_haiku_6_s2: OK — all invariants hold
- indep_opus_6: OK — all invariants hold
- indep_opus_6_iso: OK — all invariants hold
- indep_opus_6_s2: OK — all invariants hold
- indep_sonnet_6: OK — all invariants hold
- indep_sonnet_6_iso: OK — all invariants hold
- indep_sonnet_6_s2: OK — all invariants hold
- open_haiku_1: OK — all invariants hold
- open_haiku_3: OK — all invariants hold
- open_haiku_6: OK — all invariants hold
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
haiku_1             1    6   0.996598   0.994120  0.002478    20       6    22.2   0.000    10.09
haiku_3             3    2   0.996598   0.993742  0.002856    21       6    19.4   0.216    21.49
haiku_6             6    1   0.996598   0.994321  0.002277    28       6     8.3   0.199    27.37
indep_haiku_6       6    1   0.997359   0.976806  0.020553    36       1    30.6   0.146    23.33
indep_haiku_6_iso      6    1   1.012347   0.999826  0.012521    36       1    22.2   0.171     1.79
indep_haiku_6_s2      6    1   0.997333   0.995414  0.001919    36       1     0.0   0.152    16.33
indep_opus_6        6    1   0.997333   0.977649  0.019684    36       1     2.8   0.163    37.20
indep_opus_6_iso      6    1   1.012347   0.982146  0.030201    36       1     0.0   0.240    17.11
indep_opus_6_s2      6    1   0.997333   0.980071  0.017262    36       1     0.0   0.234    32.99
indep_sonnet_6      6    1   0.997333   0.988312  0.009021    36       1    11.1   0.293  2143.84
indep_sonnet_6_iso      6    1   1.012347   0.996232  0.016115    36       1    19.4   0.307     4.38
indep_sonnet_6_s2      6    1   0.997359   0.977046  0.020313    36       1    22.2   0.285  2077.04
open_haiku_1        1    1   0.996598   0.992818  0.003780    36       1     0.0   0.000     0.83
open_haiku_3        3    1   0.996598   0.990882  0.005716    36       1     0.0   0.113     4.61
open_haiku_6        6    1   0.996598   0.995085  0.001513    36       1     0.0   0.100    27.42
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

## Legacy progress table by candidate index (not an equal-compute comparison)

**Do not interpret these columns as actual GPU-attempt prefixes for rounds.** The current generator
sorts candidates by round/agent/variant, rather than GPU execution order, includes phantom candidates,
and carries scores forward after a search stops. For example, Haiku solo has 20 GPU attempts but
36 candidate entries. The `best@29/34/36` labels below are retained as legacy output, not validated
compute-matched measurements. Open records are normally one candidate per attempt, but cross-search
comparisons still require the execution timeline and the accounting caveats above. No curve or
significance claim in the revised interpretation relies on this table.

| cell | runs | best@29 | best@34 | best@36 | final |
|---|---:|---:|---:|---:|---:|
| haiku_1 | 20 | 0.996000 | 0.994120 | 0.994120 | 0.99412 |
| haiku_3 | 21 | 0.995001 | 0.993742 | 0.993742 | 0.993742 |
| haiku_6 | 28 | 0.994321 | 0.994321 | 0.994321 | 0.994321 |
| indep_haiku_6 | 36 | 0.976806 | 0.976806 | 0.976806 | 0.976806 |
| indep_haiku_6_iso | 36 | 0.999826 | 0.999826 | 0.999826 | 0.999826 |
| indep_haiku_6_s2 | 36 | 0.995414 | 0.995414 | 0.995414 | 0.995414 |
| indep_opus_6 | 36 | 0.981666 | 0.978436 | 0.977649 | 0.977649 |
| indep_opus_6_iso | 36 | 0.982146 | 0.982146 | 0.982146 | 0.982146 |
| indep_opus_6_s2 | 36 | 0.981817 | 0.980071 | 0.980071 | 0.980071 |
| indep_sonnet_6 | 36 | 0.988312 | 0.988312 | 0.988312 | 0.988312 |
| indep_sonnet_6_iso | 36 | 0.998370 | 0.996232 | 0.996232 | 0.996232 |
| indep_sonnet_6_s2 | 36 | 0.977046 | 0.977046 | 0.977046 | 0.977046 |
| open_haiku_1 | 36 | 0.993136 | 0.992818 | 0.992818 | 0.992818 |
| open_haiku_3 | 36 | 0.991949 | 0.991730 | 0.990882 | 0.990882 |
| open_haiku_6 | 36 | 0.995457 | 0.995085 | 0.995085 | 0.995085 |
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
