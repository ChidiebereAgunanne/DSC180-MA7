"""The 30-second demo that motivates the whole methodology.

Run: python -m src.leakage_demo
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.config import TARGET
from src.features import build_features
from src.ingest import load_raw
from src.train import temporal_split

if __name__ == "__main__":
    df = build_features(load_raw())
    train, _, test = temporal_split(df)

    leaky = ["casual", "registered", "temp", "hum"]
    model = LinearRegression().fit(train[leaky], train[TARGET])
    r2 = r2_score(test[TARGET], model.predict(test[leaky]))

    print(f"Features: {leaky}")
    print(f"Test R^2 = {r2:.6f}")
    print()
    print("This model is useless. casual + registered == cnt by definition,")
    print("so it has learned the identity function. Nothing here forecasts")
    print("anything: at prediction time you would not know either input.")
