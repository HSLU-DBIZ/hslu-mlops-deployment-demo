"""Step 3 · Baseline ladder: is the model worth its compute?

uv run --group efficiency python scripts/efficiency/3_ladder.py

Three years of history (fetched once, see lucerne_temp.features.history): two years to
train, the last full year to test, all 24 horizons, the 9 features of v2. Every model is
wrapped in the package's Forecaster, so all of them predict the change from temp_now.
"""

import lightgbm as lgb
from _common import DAILY, history_dataset, mae, show, size_kb, split, timed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lucerne_temp.common.model import Forecaster
from lucerne_temp.common.registry import FileModelStore

MODELS = {
    "Ridge (scaler + linear)": lambda: make_pipeline(StandardScaler(), Ridge()),
    "HistGradientBoosting, v2 params": lambda: HistGradientBoostingRegressor(
        random_state=42, **FileModelStore().version_config("2")["params"]
    ),
    "LightGBM, 400 trees": lambda: lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, n_jobs=4, random_state=0, verbose=-1
    ),
    "MLP 64-32 (small neural net)": lambda: make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, random_state=0),
    ),
}


def row(name: str, test, predicted, fit_ms: float = 0.0, kb: float = 0.0) -> dict:
    error = (test["target"] - predicted).abs()
    by_h = error.groupby(test["hours_ahead"]).mean()
    return {
        "model": name,
        "MAE": round(error.mean(), 3),
        "MAE 1h": round(by_h[1], 3),
        "MAE 6h": round(by_h[6], 3),
        "MAE 24h": round(by_h[24], 3),
        "train s": round(fit_ms / 1000, 2),
        "size KB": round(kb),
    }


def main() -> None:
    train, test = split(history_dataset())
    for name, rows in [("train", train), ("test", test)]:
        print(f"{name}: {len(rows)} rows, feature times {rows.index.min()} to {rows.index.max()}")
    rows = [
        row("Persistence (temp_now)", test, test["temp_now"]),
        row("Same hour yesterday", test, test["same_hour_yesterday"]),
    ]
    for name, make in MODELS.items():
        model = Forecaster(make(), DAILY)
        fit_ms, _ = timed(lambda: model.fit(train, train["target"]))  # noqa: B023
        rows.append(row(name, test, model.predict(test), fit_ms[0], size_kb(model)))
    show(rows, "Test year, MAE in °C (one fit each)")
    print(f"\nMAE check: {mae(test['target'], test['temp_now']):.3f} = persistence row")


if __name__ == "__main__":
    main()
