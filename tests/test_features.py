from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from lucerne_temp.features.build import build_features, serving_features
from lucerne_temp.features.ingest import load_snapshot


@pytest.mark.parametrize("hours_ahead", [1, 6, 24])
def test_features_agree(hours_ahead):
    t = datetime(2026, 9, 1, 12, tzinfo=UTC)
    raw = load_snapshot()
    # what training saw (features over the full history) vs what serving computes
    # (only the raw hours that exist at time t)
    offline = build_features(raw, hours_ahead).loc[t]
    online, valid_time, feature_time = serving_features(raw[:t], hours_ahead)
    assert feature_time == t
    assert valid_time == t + pd.Timedelta(hours=hours_ahead)
    assert np.allclose(offline, online.iloc[0])


def test_no_feature_peeks_ahead():
    raw = load_snapshot()
    t = raw.index[500]
    changed_future = raw.copy()
    changed_future[raw.index > t] += 10
    for hours_ahead in (1, 24):
        before = build_features(raw, hours_ahead).loc[t]
        after = build_features(changed_future, hours_ahead).loc[t]
        assert np.allclose(before, after)


def test_lags_are_hours_not_rows():
    raw = load_snapshot().iloc[:48]
    t = raw.index[-1]
    X = build_features(raw, hours_ahead=3).loc[t]
    assert X["lag_1"] == raw[t - pd.Timedelta(hours=1)]
    assert X["same_hour_yesterday"] == raw[t - pd.Timedelta(hours=21)]  # valid time - 24 h


def test_gap_in_raw_data_is_an_error_not_a_shifted_lag():
    raw = load_snapshot().iloc[:48]
    with pytest.raises(ValueError, match="incomplete features"):
        serving_features(raw.drop(raw.index[-2]), hours_ahead=1)
