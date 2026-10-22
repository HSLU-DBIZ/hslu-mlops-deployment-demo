"""Step 5a · Speed knobs for trees: make each fit cheap before you search.

uv run --group efficiency python scripts/efficiency/5a_knobs.py

LightGBM on the two training years, early stopping on the last 90 days of them (time
folds with gap, so no target from the validation days is trained on). Each setting is
fitted 3 times; the table shows the median wall time and the test-year MAE.
"""

import numpy as np
import pandas as pd
from _common import (
    DAILY,
    EXTENDED,
    fit_lightgbm,
    history_dataset,
    mae,
    show,
    size_kb,
    split,
    timed,
)

from lucerne_temp.features.build import BASE_FEATURES
from lucerne_temp.training.folds import time_folds

REPEATS = 3


def main() -> None:
    train, test = split(history_dataset(extended=True))
    ((fit_idx, val_idx),) = time_folds(train, n_splits=1, test_days=90)
    fit_rows, val_rows = train.iloc[fit_idx], train.iloc[val_idx]

    def run(setting: str, features=DAILY, repeats=REPEATS, **params) -> dict:
        times, model = timed(lambda: fit_lightgbm(fit_rows, val_rows, features, **params), repeats)
        return {
            "setting": setting,
            "fit s": round(float(np.median(times)) / 1000, 2),
            "trees": model.estimator.best_iteration_ or model.estimator.n_estimators,
            "test MAE": round(mae(test["target"], model.predict(test)), 3),
            "size KB": round(size_kb(model)),
        }

    rows = [run("5000 trees, no early stopping", repeats=1, patience=None)]
    rows += [run(f"early stopping, n_jobs={n}", n_jobs=n) for n in (1, 2, 4, 8, 18)]
    rows += [
        run("num_leaves=15", num_leaves=15),
        run("num_leaves=7", num_leaves=7),
        run("colsample_bytree=0.5", colsample_bytree=0.5),
        run(f"{len(EXTENDED)} features (+lags, rolling, day of year)", EXTENDED),
        run(f"{len(BASE_FEATURES)} features (v1)", BASE_FEATURES),
    ]
    show(rows, f"LightGBM, lr 0.05, n_jobs=4 unless stated; fit = {len(fit_rows)} rows")

    gain = pd.Series(
        fit_lightgbm(fit_rows, val_rows, EXTENDED).estimator.booster_.feature_importance("gain"),
        index=EXTENDED,
    ).sort_values(ascending=False)
    print("\nTop 10 of the extended features by gain:", ", ".join(gain.index[:10]))


if __name__ == "__main__":
    main()
