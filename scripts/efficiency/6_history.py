"""Step 6 · How much history is enough? A learning curve over the length of history.

uv run --group efficiency python scripts/efficiency/6_history.py

Train on the last 1, 3, 6, 12 or 24 months before the test year, score on the same test
year. LightGBM stops early on the last fifth of its window (time folds with gap).
"""

import numpy as np
from _common import DAILY, TEST_START, fit_lightgbm, history_dataset, mae, show, split, timed
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lucerne_temp.common.model import Forecaster
from lucerne_temp.training.folds import time_folds

MONTHS = [1, 3, 6, 12, 24]


def main() -> None:
    train, test = split(history_dataset())
    rows = []
    for months in MONTHS:
        window = train[train.index >= TEST_START - np.timedelta64(months * 30, "D")]
        ridge = Forecaster(make_pipeline(StandardScaler(), Ridge()), DAILY)
        ridge_ms, _ = timed(lambda: ridge.fit(window, window["target"]), repeats=3)  # noqa: B023
        ((a, b),) = time_folds(window, n_splits=1, test_days=months * 6)  # last fifth
        lgbm_ms, lgbm = timed(lambda: fit_lightgbm(window.iloc[a], window.iloc[b], DAILY))  # noqa: B023
        rows.append(
            {
                "months": months,
                "from": f"{window.index.min():%Y-%m-%d}",
                "rows": len(window),
                "Ridge MAE": round(mae(test["target"], ridge.predict(test)), 3),
                "Ridge ms": round(float(np.median(ridge_ms))),
                "LightGBM MAE": round(mae(test["target"], lgbm.predict(test)), 3),
                "LightGBM s": round(lgbm_ms[0] / 1000, 2),
                "trees": lgbm.estimator.best_iteration_,
            }
        )
    show(rows, f"Test year from {TEST_START:%Y-%m-%d}, MAE in °C; a month = 30 days")


if __name__ == "__main__":
    main()
