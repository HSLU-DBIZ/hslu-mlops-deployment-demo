"""Shared by the scripts in scripts/efficiency/: data split, extra features, timing, tables.

Features come from the package's build_features(), rows from make_dataset(), metrics from
evaluate(): the experiments measure the pipeline that ships, not a copy of it.
"""

import io
import time
from collections.abc import Callable

import joblib
import numpy as np
import pandas as pd

from lucerne_temp.common.model import Forecaster
from lucerne_temp.features.build import FEATURE_SETS
from lucerne_temp.features.history import load_history
from lucerne_temp.training.train import make_dataset

# three years of history: two to train on, the last full year to test on (every season)
TEST_START = pd.Timestamp("2025-10-01", tz="UTC")
DAILY = FEATURE_SETS["daily"]
EXTRA = [
    *(f"lag_{k}" for k in range(4, 24)), "roll_mean_24", "roll_mean_168", "doy_sin", "doy_cos",
]  # fmt: skip
EXTENDED = [*DAILY, *EXTRA]


def history_dataset(extended: bool = False) -> pd.DataFrame:
    """make_dataset() on the three-year history; extended adds 24 more columns."""
    temps = load_history()
    data = make_dataset(temps)
    if extended:
        data = data.join(extra_features(temps)).dropna()
    return data


def extra_features(temps: pd.Series) -> pd.DataFrame:
    """Columns on top of build_features(); like it, they only use values at or before t."""
    temps = temps.asfreq("h")
    doy = temps.index.dayofyear
    return pd.DataFrame(
        {
            **{f"lag_{k}": temps.shift(k) for k in range(4, 24)},
            "roll_mean_24": temps.rolling(24).mean(),
            "roll_mean_168": temps.rolling(168).mean(),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        }
    )


def split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time split as in train.py: no training target lies in the test year."""
    train = data[data["valid_time"] < TEST_START]
    test = data[data.index >= TEST_START]
    return train, test


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def size_kb(obj) -> float:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return buffer.tell() / 1024


def timed(fn: Callable, repeats: int = 1, warmup: int = 0) -> tuple[np.ndarray, object]:
    """Wall times in ms of `repeats` calls after `warmup` untimed calls, and the last result."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times), result


def show(rows: list[dict], title: str = "") -> None:
    if title:
        print(f"\n{title}")
    table = pd.DataFrame(rows)
    for col in table.columns[table.isna().any()]:  # blank cells instead of NaN, 16 not 16.0
        whole = all(
            isinstance(v, int | float) and float(v).is_integer() for v in table[col].dropna()
        )
        table[col] = ["" if pd.isna(v) else int(v) if whole else v for v in table[col]]
    print(table.to_string(index=False))


def fit_lightgbm(
    fit_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    features: list[str],
    patience: int | None = 100,
    callbacks: tuple = (),
    **params,
):
    """LightGBM in a Forecaster, stopped early on val_rows' MAE (patience=None: never).

    Forecaster.fit() takes no extra arguments, so the estimator is fitted directly, on the
    same target Forecaster uses: the change from temp_now.
    """
    import lightgbm as lgb  # only the efficiency group installs it

    params = {
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "n_jobs": 4,
        "random_state": 0,
        "verbose": -1,
        **params,
    }
    estimator = lgb.LGBMRegressor(**params)
    estimator.fit(
        fit_rows[features],
        fit_rows["target"] - fit_rows["temp_now"],
        eval_X=val_rows[features],
        eval_y=val_rows["target"] - val_rows["temp_now"],
        eval_metric="l1",
        callbacks=[
            *callbacks,
            *([lgb.early_stopping(patience, verbose=False)] if patience else []),
        ],
    )
    return Forecaster(estimator, features)
