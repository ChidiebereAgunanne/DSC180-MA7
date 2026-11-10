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


def test_no_overlapping_dates(splits):
    train, valid, test = splits
    dates = pd.concat([train.dteday, valid.dteday, test.dteday])
    assert dates.is_unique
