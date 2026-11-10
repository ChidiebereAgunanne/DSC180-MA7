"""Stage 2: build the modelling table.

The contract this module guarantees, and that tests/ enforces:
  1. No column in FORBIDDEN (see src/config.py) reaches the model.
  2. Every feature at time t is computable from information available at t.
  3. The output is sorted by date, gap-free after dropna, and complete.

Run: python -m src.features
"""
import numpy as np
import pandas as pd

from src.config import FORBIDDEN, LAG_WARMUP, PROCESSED, TARGET
from src.ingest import load_raw

WEATHER_ORDER = {"GOOD": 0, "MISTY": 1, "BAD": 2}


def encode_weather(series: pd.Series) -> pd.Series:
    """Ordinal-encode weather, failing loudly on unseen labels.

    A silent `.map()` turns an unexpected label into NaN, which a later
    `dropna()` deletes without comment. That is how the 20 BAD-weather
    days -- the most interesting days in the dataset -- disappear with no
    error and no warning.
    """
    known = series.dropna()
    unknown = set(known.unique()) - set(WEATHER_ORDER)
    if unknown:
        raise ValueError(f"unmapped weather labels: {sorted(unknown)}")
    return series.map(WEATHER_ORDER).astype(float)


def reindex_to_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Insert a row for every missing calendar day.

    This mirror is missing 2011-03-10. pandas `.shift(1)` moves by rows,
    not by days, so without this step the row after the gap gets a two-day
    lag labelled as a one-day lag, `.shift(7)` is off by a day one week
    later, and the 7-day rolling mean is contaminated for a week. Nothing
    in the data announces any of it. After reindexing, the gap is an
    explicit NaN that `dropna()` removes, and every lag means what it says.
    """
    full = pd.date_range(df.dteday.min(), df.dteday.max(), freq="D")
    return (
        df.set_index("dteday")
        .reindex(full)
        .rename_axis("dteday")
        .reset_index()
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = reindex_to_calendar(df.sort_values("dteday").reset_index(drop=True)).copy()

    # --- Calendar: known in advance, so it cannot leak -------------------
    out["dayofweek"] = out.dteday.dt.dayofweek
    out["is_weekend"] = (out.dayofweek >= 5).astype(int)
    out["is_holiday"] = (out.holiday == "Y").astype(int)
    out["is_workday"] = (out.workday == "Y").astype(int)

    # Cyclical month encoding, so December and January sit next to each
    # other instead of 11 units apart.
    month = out.dteday.dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)

    # --- Weather: measured same-day, available to a forecaster ----------
    out["weather_severity"] = encode_weather(out.weather)

    # --- Lagged target: past values only. shift(1) is the whole ballgame;
    # without it, cnt_roll_7 would contain today's count. -----------------
    # If you add a longer lag, raise LAG_WARMUP in src/config.py.
    out["cnt_lag_1"] = out[TARGET].shift(1)
    out["cnt_lag_7"] = out[TARGET].shift(LAG_WARMUP)
    out["cnt_roll_7"] = out[TARGET].shift(1).rolling(LAG_WARMUP).mean()

    return out.dropna().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are safe to feed the model."""
    numeric = df.select_dtypes(include="number").columns
    return [c for c in numeric if c not in FORBIDDEN]


if __name__ == "__main__":
    feats = build_features(load_raw())
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(PROCESSED, index=False)
    cols = feature_columns(feats)
    print(f"OK features: {len(feats)} rows, {len(cols)} features -> {PROCESSED.name}")
    print("  " + ", ".join(cols))
