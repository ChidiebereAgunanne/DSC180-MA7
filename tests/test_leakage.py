"""The tests that make the agent honest.

test_no_forbidden_features is the single most valuable test in this repo:
casual + registered == cnt exactly, so any model handed those columns
scores R^2 = 1.0 and looks brilliant while having learned nothing.
"""
import pandas as pd
import pytest

from src.config import FORBIDDEN, TARGET
from src.features import build_features, feature_columns
from src.ingest import load_raw


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw())


def test_leakage_identity_holds(feats):
    """Documents *why* casual and registered are forbidden."""
    assert (feats.casual + feats.registered == feats[TARGET]).all()


def test_no_forbidden_features(feats):
    """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
    construction, so this cannot discover a *new* leaky column. What it
    catches is someone editing feature_columns() or FORBIDDEN -- which is
    exactly the fault the Step 7 hook demonstrates."""
    cols = set(feature_columns(feats))
    assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"


def test_lags_are_true_calendar_lags(feats):
    """cnt_lag_1 for date d must equal cnt on d-1, in calendar days."""
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), f"bad lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_lag_is_not_the_target(feats):
    assert not (feats.cnt_lag_1 == feats[TARGET]).all()


def test_rolling_mean_excludes_today(feats):
    """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
    by_date = feats.set_index("dteday")[TARGET]
    date = feats.dteday.iloc[200]
    window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
    if all(d in by_date.index for d in window):
        expected = sum(by_date[d] for d in window) / 7
        assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)


def test_cnt_change_1d_hand_calculated(feats):
    """cnt_change_1d on d is cnt[d-1] - cnt[d-2], both strictly before d.

    Hand-computed from data/raw/bike_daily.csv:
      2012-03-13 cnt = 5847, 2012-03-14 cnt = 6312, 2012-03-15 cnt = 6192
      expected on 2012-03-15: 6312 - 5847 = 465
    A same-day leak (cnt.diff() with no shift) would give 6192 - 6312 = -120.
    """
    row = feats.set_index("dteday").loc[pd.Timestamp("2012-03-15")]
    assert row.cnt_change_1d == pytest.approx(465)
    assert row[TARGET] == 6192  # today's count is not an input


def test_cnt_change_1d_is_a_calendar_difference(feats):
    """Every surviving row must use the two preceding *calendar* days. The
    mirror lacks 2011-03-10, so a row-based diff would pair 2011-03-11 with
    2011-03-09 (1977 - 1891 = 86) and label it a one-day change."""
    by_date = feats.set_index("dteday")
    raw = load_raw().set_index("dteday")[TARGET]
    checked = 0
    for date, change in by_date.cnt_change_1d.items():
        d1, d2 = date - pd.Timedelta(days=1), date - pd.Timedelta(days=2)
        assert d1 in raw.index and d2 in raw.index, f"{date.date()} spans a calendar gap"
        assert change == pytest.approx(raw[d1] - raw[d2]), f"bad change at {date.date()}"
        checked += 1
    assert checked == len(feats)


def test_no_missing_values(feats):
    assert feats[feature_columns(feats)].isna().sum().sum() == 0


def test_dates_are_sorted_and_unique(feats):
    assert feats.dteday.is_monotonic_increasing
    assert feats.dteday.is_unique
