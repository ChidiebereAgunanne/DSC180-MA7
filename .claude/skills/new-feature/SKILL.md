---
name: new-feature
description: Add a feature column to the modelling table with its leakage test. Use whenever someone asks to add, engineer, or try a new predictor.
allowed-tools: Read, Edit, Bash(python -m pytest*), Bash(make*)
---

# Current feature set

!`python -c "import pandas as pd; from src.features import feature_columns; print(', '.join(feature_columns(pd.read_parquet('data/processed/features.parquet'))))"`

# Adding a feature: $ARGUMENTS

Work in this order. Do not skip step 2.

1. **State the availability argument in one sentence.** At prediction time
   for date *d*, is every input to this feature known? If it depends on
   anything measured on or after *d*, say so and stop.

2. **Write the failing test first** in `tests/test_leakage.py`. For a lag
   or rolling feature, assert its value on a known date equals the
   hand-computed value from strictly earlier dates. Run `make test` and
   confirm the new test fails for the right reason.

3. **Implement it** in `src.features.build_features`, after the
   `reindex_to_calendar` call. Any aggregation gets `.shift(1)` before
   `.rolling()`. If the feature needs more than 7 days of history, raise
   `LAG_WARMUP` in `src/config.py` so the row-loss test stays honest.

4. **Verify.** Run `make features && make test && make train`. Report the
   before/after validation R^2 for ridge and hgb from
   `reports/metrics.json`. Do not report test-set numbers while iterating.

5. **Record the decision** as one line in `docs/DECISIONS.md`: what you
   added, the availability argument, and the validation delta.

If validation R^2 improves by less than 0.01, say so plainly and
recommend dropping the feature. The validation set is 182 days; a change
that small is inside the noise, not a gain.
