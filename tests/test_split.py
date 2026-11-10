"""Splits must be temporal, disjoint, and non-empty."""
import pandas as pd
import pytest

from src.config import TRAIN_END, VALID_END
from src.features import build_features
from src.ingest import load_raw
from src.train import temporal_split


@pytest.fixture(scope="module")
def splits():
    return temporal_split(build_features(load_raw()))


def test_splits_non_empty(splits):
    for part in splits:
        assert len(part) > 0


def test_splits_are_ordered_in_time(splits):
    train, valid, test = splits
    assert train.dteday.max() < valid.dteday.min()
    assert valid.dteday.max() < test.dteday.min()


def test_valid_only_path_never_reads_test_rows():
    """Poison every test-period value with NaN. If score_valid_only fit on
    or scored any test row, the models would raise or the scores would
    move. Identical results prove the trial path is blind to test."""
    from src.features import feature_columns
    from src.train import score_valid_only

    df = build_features(load_raw())
    cols = feature_columns(df)
    clean = score_valid_only(df, cols)

    poisoned = df.copy()
    after = poisoned.dteday > VALID_END
    poisoned.loc[after, cols + ["cnt"]] = float("nan")
    assert score_valid_only(poisoned, cols) == clean
    assert "test" not in str(clean)


def test_no_overlapping_dates(splits):
    train, valid, test = splits
    dates = pd.concat([train.dteday, valid.dteday, test.dteday])
    assert dates.is_unique
