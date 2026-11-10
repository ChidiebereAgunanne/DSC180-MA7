"""Single source of truth for paths, columns, and split dates."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bike_daily.csv"
PROCESSED = ROOT / "data" / "processed" / "features.parquet"
METRICS = ROOT / "reports" / "metrics.json"
# Validation-only scores for feature trials (python -m src.train --valid-only).
# Never contains test-set numbers; METRICS is written only by the final stage.
METRICS_VALID = ROOT / "reports" / "metrics_valid.json"

# Daily Capital Bikeshare counts (Fanaee-T & Gama). Canonical home:
#   https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
# We use the processed copy from Molnar's "Interpretable Machine Learning"
# because the UCI host blocks some campus networks. It differs from UCI:
#   - 728 rows, 2011-01-03 to 2012-12-31 (UCI has 731 from 2011-01-01)
#   - 2011-03-10 is absent (the UCI row records humidity 0, a sensor
#     artefact, and the mirror's cleaning dropped the whole day)
#   - `weather` is the labels GOOD / MISTY / BAD, not UCI's codes 1-4
#   - two extra columns: `days_since_2011` and `cnt_2d_bfr`
DATA_URL = (
    "https://raw.githubusercontent.com/christophM/"
    "interpretable-ml-book/master/data/bike.csv"
)

TARGET = "cnt"

# --- Three reasons a column is kept out of the feature set --------------

# 1. Leakage: the target or an arithmetic restatement of it.
#    casual + registered == cnt exactly, on every row. A model handed these
#    scores R^2 ~ 1.0 and has learned addition. At prediction time you would
#    not know either input: they are components of the thing being predicted.
LEAKAGE_COLS = {"casual", "registered", "cnt"}

# 2. Time indexes: not leakage, but a bad bet. A tree splits on "after date
#    X" and predicts a flat constant for every future date; a linear model
#    extrapolates a two-year trend we have no reason to trust. Excluding
#    them keeps the model comparison about what each does with lagged demand.
TIME_INDEX_COLS = {"instant", "yr", "days_since_2011"}

# 3. Unverified: `cnt_2d_bfr` is a two-day lag of cnt that shipped with the
#    mirror. It is NOT leakage -- a two-day-old count is known at prediction
#    time -- but we did not build it and cannot test how it was aligned, so we
#    build our own lags from the target and test those instead.
UNVERIFIED_COLS = {"cnt_2d_bfr"}

# What feature_columns() and the tests actually enforce.
FORBIDDEN = LEAKAGE_COLS | TIME_INDEX_COLS | UNVERIFIED_COLS

# --- Every raw column, classified by hand ---------------------------------
# A numeric dtype is not an approval. src.ingest.load_raw raises on any raw
# column missing from this map; tests/test_contract.py checks each column
# sits in exactly one class. Adding a raw column means deciding its class.

# Same-day weather readings: measured on d, treated as known to a
# forecaster (a next-day weather forecast stands in for them in deployment).
APPROVED_RAW_FEATURES = {"temp", "atemp", "hum", "windspeed"}

# Inputs to engineered features in src/features.py, never fed raw: the
# date drives dayofweek/month_sin/month_cos; labels are encoded explicitly.
# season/mnth/weekday duplicate what the date already gives.
ENCODED_ONLY_COLS = {"dteday", "season", "mnth", "weekday", "holiday", "workday", "weather"}

RAW_COLUMN_CLASSES = {
    "leakage": LEAKAGE_COLS,
    "time_index": TIME_INDEX_COLS,
    "unverified": UNVERIFIED_COLS,
    "approved_feature": APPROVED_RAW_FEATURES,
    "encoded_only": ENCODED_ONLY_COLS,
}
CLASSIFIED_RAW_COLS = set().union(*RAW_COLUMN_CLASSES.values())

# Temporal split: train on 2011, validate on H1 2012, test on H2 2012.
TRAIN_END = "2011-12-31"
VALID_END = "2012-06-30"

# Longest lag window in src/features.py. The first LAG_WARMUP calendar days
# have no complete history and are dropped; tests/test_contract.py uses this
# to bound how many rows may legitimately disappear.
LAG_WARMUP = 7
