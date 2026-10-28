# Methodology

## Dataset Health Score (0-100, project-defined heuristic)
score = Σ(wᵈ·sᵈ) / Σ wᵈ over *applicable* dimensions. Default weights: completeness .25, uniqueness .15, validity .25,
statistical .15, label quality .20 (only with a class target; otherwise its weight is redistributed, never scored as 100).

| Dimension | Formula |
|---|---|
| Completeness | 0.7·100(1−min(1, 2·missing-cell fraction)) + 0.3·(share of columns with < excessive-missing threshold) |
| Uniqueness | 100(1−min(1, 3·duplicate-row fraction)) |
| Validity | 100(1 − flagged columns / columns); flagged = numeric-as-text, mixed types, date-as-text, inconsistent labels |
| Statistical | 100(1−min(1, 3·outlier-cell fraction)) − 5/redundant pair (max 25) − 5/constant column (max 15) |
| Label quality | 100 if imbalance ratio ≤ 1.5, falling log-linearly to 20 at ratio ≥ 20 |

Categories (project-defined, configurable): 90 Excellent · 75 Good · 60 Needs Improvement · 40 Poor · 0 Critical.
Fewer than 5 rows → no score ("Insufficient data"). The score is not a probability, accuracy estimate or ML-performance guarantee.
Each underlying issue feeds exactly one dimension. Leakage and bias indicators are *not* scored.

## Severity rules
- Missing: ≥ 40 % critical (configurable) · ≥ 20 % high · ≥ 5 % medium · else low. Missing target values: high (≥1 %) / medium.
- Duplicates: ≥ 20 % high · ≥ 5 % medium · else low.
- Outliers: ≥ 10 % medium · ≥ 1 % low · else info (an outlier is not necessarily an error).
- Class imbalance (majority/minority): ≥ 20 critical · ≥ 10 high · ≥ 3 medium · ≥ 1.5 low.
- Inconsistent labels: high on the target column, else by affected share (≥ 20 high · ≥ 5 medium · else low).
- Leakage indicators: high when name *and/or* very strong association (η or |r| or Cramér's V ≥ 0.95) with the target; medium for name-only. Always "potential".

## ML-readiness
Checks: target availability, missing values, data types, duplicates, class distribution, feature quality, leakage indicators,
label consistency, unresolved critical issues. Insufficient information if < 30 rows or no target for a supervised purpose;
"Requires substantial cleaning" if any critical issue is open or ≥ 4 checks fail; "Ready after recommended fixes" if some fail;
"Ready for initial experimentation" if none fail.

## Known limitations
Bias/leakage checks are indicators with stated limits; no semantic label validation; statistics for imputation are computed on the full
dataset (fit on training data only in real ML pipelines); resampling should be applied to training splits only.
