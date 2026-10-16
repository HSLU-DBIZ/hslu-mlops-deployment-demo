"""Training pipeline: python -m lucerne_temp.training.train --version 1

Reads the committed snapshot, builds features with the shared feature function, trains
the version described in registry/registry.json and saves the artifact to artifacts/.
Deterministic: same snapshot + same config = same model.
"""

import argparse
import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from lucerne_temp.common.model import Forecaster
from lucerne_temp.common.registry import FileModelStore
from lucerne_temp.features.build import FEATURE_SETS, build_features
from lucerne_temp.features.ingest import load_snapshot

HORIZONS = range(1, 25)
TEST_DAYS = 14
SEED = 42


def make_dataset(temps: pd.Series) -> pd.DataFrame:
    """Feature rows for every hour and every horizon, plus the observed target."""
    temps = temps.asfreq("h")
    frames = []
    for h in HORIZONS:
        X = build_features(temps, h)
        X["valid_time"] = X.index + pd.Timedelta(hours=h)
        X["target"] = temps.shift(-h)
        frames.append(X)
    return pd.concat(frames).dropna()


def evaluate(model: Forecaster, data: pd.DataFrame) -> dict:
    """MAE of the model next to the persistence baseline (forecast = last observed value)."""
    error = (data["target"] - model.predict(data)).abs()
    baseline = (data["target"] - data["temp_now"]).abs()
    by_h = pd.DataFrame({"model": error, "persistence": baseline}).groupby(data["hours_ahead"])
    return {
        "mae": round(error.mean(), 3),
        "mae_persistence": round(baseline.mean(), 3),
        "mae_by_horizon": {
            str(h): by_h.mean().round(3).loc[h].to_dict() for h in (1, 3, 6, 12, 24)
        },
    }


def train(version: str, store: FileModelStore | None = None, register: bool = False) -> dict:
    store = store or FileModelStore()
    cfg = store.version_config(version)
    data = make_dataset(load_snapshot())

    # out-of-time split: the last TEST_DAYS are never seen in training
    split = data.index.max() - pd.Timedelta(days=TEST_DAYS)
    train_rows = data[data["valid_time"] <= split]
    test_rows = data[data.index > split]

    estimator = HistGradientBoostingRegressor(random_state=SEED, **cfg["params"])
    model = Forecaster(estimator, FEATURE_SETS[cfg["feature_set"]])
    model.fit(train_rows, train_rows["target"])

    info = {
        "train_window": [str(train_rows.index.min()), str(split)],
        "test_window": [str(test_rows.index.min()), str(test_rows.index.max())],
        "metrics": evaluate(model, test_rows),
    }
    store.save(version, model)
    if register:
        store.record(version, **info)
    return info


def ensure_trained(version: str, store: FileModelStore | None = None) -> Forecaster:
    """Load the artifact; rebuild it from the snapshot if it is not there yet."""
    store = store or FileModelStore()
    if not store.artifact_path(version).exists():
        print(f"artifact for v{version} missing: training from the snapshot")
        train(version, store)
    return store.load(version)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="registry version to train (default: champion)")
    parser.add_argument("--register", action="store_true", help="write metrics to the registry")
    args = parser.parse_args()
    store = FileModelStore()
    version = args.version or store.version_of()
    print(json.dumps({"version": version, **train(version, store, args.register)}, indent=2))


if __name__ == "__main__":
    main()
