"""Step 4 · Honest validation: which number would you have believed?

uv run --group efficiency python scripts/efficiency/4_validation.py

The same LightGBM model scored five ways on the two training years, next to the test
year it never saw. Then the same check for preprocessing: a scaler fitted on all rows
before the split versus inside a Pipeline.
"""

import lightgbm as lgb
import numpy as np
from _common import DAILY, history_dataset, mae, show, split
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lucerne_temp.common.model import Forecaster
from lucerne_temp.training.folds import time_folds

FOLDS, TEST_DAYS = 5, 60


def lightgbm() -> Forecaster:
    estimator = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, n_jobs=4, random_state=0, verbose=-1
    )
    return Forecaster(estimator, DAILY)


def cv(make, data, folds) -> float:
    scores = []
    for fit_rows, test_rows in folds:
        a, b = data.iloc[fit_rows], data.iloc[test_rows]
        scores.append(mae(b["target"], make().fit(a, a["target"]).predict(b)))
    return float(np.mean(scores))


def main() -> None:
    train, test = split(history_dataset())
    in_time_order = train.sort_index(kind="stable")  # make_dataset stacks the horizons
    holdout = mae(test["target"], lightgbm().fit(train, train["target"]).predict(test))
    schemes = {
        "random KFold, shuffled": (
            train,
            list(KFold(FOLDS, shuffle=True, random_state=0).split(train)),
        ),
        "TimeSeriesSplit, rows in time order, no gap": (
            in_time_order,
            list(TimeSeriesSplit(FOLDS, test_size=TEST_DAYS * 24 * 24).split(in_time_order)),
        ),
        "time folds, no gap": (train, time_folds(train, FOLDS, TEST_DAYS, gap=False)),
        "time folds, gap = horizon": (train, time_folds(train, FOLDS, TEST_DAYS)),
    }
    rows = [
        {"scheme": name, "CV MAE": round(cv(lightgbm, data, folds), 3)}
        for name, (data, folds) in schemes.items()
    ]
    rows.append({"scheme": "holdout: the test year", "CV MAE": round(holdout, 3)})
    show(rows, f"LightGBM, {FOLDS} folds ({TEST_DAYS} days each for the time-ordered ones)")

    # preprocessing: a scaler fitted once on every training row (test folds included),
    # frozen, versus one refitted inside each fold by the Pipeline
    folds = time_folds(train, FOLDS, TEST_DAYS)
    scaler_all = StandardScaler().fit(train[DAILY])
    leaky = lambda: Forecaster(make_pipeline(FrozenEstimator(scaler_all), Ridge()), DAILY)  # noqa: E731
    clean = lambda: Forecaster(make_pipeline(StandardScaler(), Ridge()), DAILY)  # noqa: E731
    show(
        [
            {
                "Ridge": "scaler fitted before the split",
                "CV MAE": round(cv(leaky, train, folds), 4),
            },
            {"Ridge": "scaler inside the Pipeline", "CV MAE": round(cv(clean, train, folds), 4)},
        ],
        "Preprocessing inside or outside the folds (time folds, gap = horizon)",
    )


if __name__ == "__main__":
    main()
