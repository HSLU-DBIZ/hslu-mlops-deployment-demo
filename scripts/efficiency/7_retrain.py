"""Step 7 · Retraining policy: what does each policy buy, and what does it cost?

uv run --group efficiency python scripts/efficiency/7_retrain.py

Replays the test year month by month. At the start of every month a policy may retrain
on what is known by then (rows whose target lies before the month), then predicts the
month. LightGBM with fixed settings (400 trees, lr 0.05) so the policies differ only in
what they train on.
"""

import lightgbm as lgb
import pandas as pd
from _common import DAILY, TEST_START, history_dataset, show, size_kb, timed

from lucerne_temp.common.model import Forecaster

PARAMS = {"n_estimators": 400, "learning_rate": 0.05, "n_jobs": 4, "random_state": 0, "verbose": -1}
WARM_TREES = 50  # trees added per month by the warm-start policy


def fit(rows: pd.DataFrame, init: Forecaster | None = None, **params) -> Forecaster:
    estimator = lgb.LGBMRegressor(**{**PARAMS, **params})
    target = rows["target"] - rows["temp_now"]  # Forecaster's target: the change
    estimator.fit(rows[DAILY], target, init_model=init.estimator.booster_ if init else None)
    return Forecaster(estimator, DAILY)


def known(data: pd.DataFrame, start: pd.Timestamp, months: int | None = None) -> pd.DataFrame:
    """Rows whose target was observed before start; optionally only the last `months`."""
    rows = data[data["valid_time"] < start]
    return rows if months is None else rows[rows.index >= start - pd.DateOffset(months=months)]


def main() -> None:
    data = history_dataset()
    months = pd.date_range(TEST_START, periods=12, freq="MS")
    policies = {
        "never (train once, before the year)": None,
        "monthly, full refit on all history": lambda start, prev: fit(known(data, start)),
        "monthly, sliding window of 12 months": lambda start, prev: fit(known(data, start, 12)),
        f"monthly, warm start (+{WARM_TREES} trees on last month)": lambda start, prev: fit(
            known(data, start, 1), init=prev, n_estimators=WARM_TREES
        ),
        f"monthly, warm start (+{WARM_TREES} trees, lr 0.01)": lambda start, prev: fit(
            known(data, start, 1), init=prev, n_estimators=WARM_TREES, learning_rate=0.01
        ),
    }
    first_ms, first = timed(lambda: fit(known(data, TEST_START)))
    rows = []
    for name, retrain in policies.items():
        model, errors, fit_ms = first, [], 0.0
        for start in months:
            if retrain and start > TEST_START:
                ms, model = timed(lambda: retrain(start, model))  # noqa: B023
                fit_ms += ms[0]
            month = data[(data.index >= start) & (data.index < start + pd.DateOffset(months=1))]
            errors.append((month["target"] - model.predict(month)).abs())
        error = pd.concat(errors)
        rows.append(
            {
                "policy": name,
                "MAE year": round(error.mean(), 3),
                "MAE last quarter": round(error[error.index >= months[9]].mean(), 3),
                "retrains": 11 if retrain else 0,
                "retrain s total": round(fit_ms / 1000, 1),
                "final size KB": round(size_kb(model)),
            }
        )
    show(rows, f"Test year by month; first model: {first_ms[0] / 1000:.1f} s on 2 years")


if __name__ == "__main__":
    main()
