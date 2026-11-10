"""Guard rails on the ingest contract and the trained model's behaviour."""
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from src.config import LAG_WARMUP, TARGET
from src.features import build_features, feature_columns, reindex_to_calendar
from src.ingest import load_raw
from src.train import MODELS, temporal_split


@pytest.fixture(scope="module")
def raw():
    return load_raw().sort_values("dteday").reset_index(drop=True)


@pytest.fixture(scope="module")
def feats(raw):
    return build_features(raw)


def test_calendar_is_complete_before_lagging(raw):
    """The reindexed frame must have no gaps, or the lags are wrong."""
    grid = reindex_to_calendar(raw)
    assert set(grid.dteday.diff().dropna().dt.days.unique()) == {1}


def test_only_expected_rows_are_dropped(raw, feats):
    """Rows may only vanish for the lag warm-up or downstream of a genuinely
    absent day -- never because an encoder produced a silent NaN.

    Accounting: the first LAG_WARMUP days have no complete history. Each
    absent calendar day m costs the LAG_WARMUP rows after it (cnt_lag_1 is
    NaN on m+1, cnt_roll_7 is NaN on m+1..m+7, cnt_lag_7 on m+7; their
    union is 7 rows). Day m itself was never in `raw`, so it is not a loss.
    The bound is exact for the current feature set; add a longer lag and
    LAG_WARMUP in src/config.py must grow with it.
    """
    full = pd.date_range(raw.dteday.min(), raw.dteday.max(), freq="D")
    missing_days = len(full) - len(raw)
    allowed_loss = LAG_WARMUP + LAG_WARMUP * missing_days
    lost = len(raw) - len(feats)
    assert lost <= allowed_loss, f"lost {lost} rows; expected at most {allowed_loss}"


def test_every_raw_column_is_explicitly_classified():
    """Each column in the raw CSV must appear in exactly one class in
    src/config.py RAW_COLUMN_CLASSES. Approval is a decision recorded in
    config, never inferred from dtype or from feature_columns() output."""
    from src.config import RAW, RAW_COLUMN_CLASSES

    header = pd.read_csv(RAW, nrows=0).columns
    for col in header:
        homes = [k for k, cols in RAW_COLUMN_CLASSES.items() if col in cols]
        assert len(homes) == 1, f"raw column {col!r} classified in {homes or 'nothing'}"


def test_unclassified_raw_column_is_rejected(tmp_path, monkeypatch):
    """A new numeric column in the raw file must stop the pipeline, not
    flow silently into the model because select_dtypes calls it a number."""
    import src.ingest
    from src.config import RAW

    df = pd.read_csv(RAW)
    df["new_sensor"] = 1.0  # numeric, not forbidden, never reviewed
    fake = tmp_path / "bike_daily.csv"
    df.to_csv(fake, index=False)
    monkeypatch.setattr(src.ingest, "RAW", fake)
    with pytest.raises(ValueError, match="new_sensor"):
        src.ingest.load_raw()


def test_raw_features_are_only_the_approved_ones(raw, feats):
    """The raw columns that reach the model must be a subset of the
    explicitly approved class, whatever their dtype."""
    from src.config import RAW_COLUMN_CLASSES

    raw_in_model = set(feature_columns(feats)) & set(raw.columns)
    assert raw_in_model <= RAW_COLUMN_CLASSES["approved_feature"]


def test_beats_the_mean_baseline(feats):
    """A model that cannot beat predicting the training mean is not a result."""
    cols = feature_columns(feats)
    train, _, test = temporal_split(feats)
    ridge = MODELS["ridge"]
    ridge.fit(train[cols], train[TARGET])
    assert r2_score(test[TARGET], ridge.predict(test[cols])) > 0.4


def test_persistence_baseline_is_reported():
    """The baseline that matters for a lag-driven model is 'yesterday again'.
    It must be in the comparison, whatever its score."""
    assert "baseline_persistence" in MODELS
