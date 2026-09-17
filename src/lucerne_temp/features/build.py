"""Feature definitions. The only place where features are computed.

Training, the batch job and the API all import build_features, so the model sees the
same features at serving time as at training time (no training-serving skew).
"""

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "temp_now", "lag_1", "lag_2", "lag_3", "roll_mean_6", "hour_sin", "hour_cos", "hours_ahead",
]  # fmt: skip
FEATURE_SETS = {
    "base": BASE_FEATURES,
    "daily": [*BASE_FEATURES, "same_hour_yesterday"],
}


def build_features(temps: pd.Series, hours_ahead: int) -> pd.DataFrame:
    """One row per feature time t, for a forecast of t + hours_ahead.

    temps: hourly temperature in °C with a UTC DatetimeIndex. A row only uses values
    at or before its own time t, so nothing peeks ahead.
    """
    temps = temps.asfreq("h")  # a missing hour becomes NaN instead of shifting the lags
    hour = temps.index.hour
    return pd.DataFrame(
        {
            "temp_now": temps,
            "lag_1": temps.shift(1),
            "lag_2": temps.shift(2),
            "lag_3": temps.shift(3),
            "roll_mean_6": temps.rolling(6).mean(),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "hours_ahead": hours_ahead,
            # temperature 24 h before the valid time; known because hours_ahead <= 24
            "same_hour_yesterday": temps.shift(24 - hours_ahead),
        }
    )


def serving_features(
    temps: pd.Series, hours_ahead: int
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """The newest feature row: (X, valid_time, feature_time)."""
    X = build_features(temps, hours_ahead).iloc[[-1]]
    if X.isna().any(axis=None):
        raise ValueError(f"incomplete features at {X.index[0]}: need 24 gap-free hours")
    feature_time = X.index[0]
    return X, feature_time + pd.Timedelta(hours=hours_ahead), feature_time
