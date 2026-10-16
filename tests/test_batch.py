import pandas as pd

from lucerne_temp.common import config
from lucerne_temp.inference import batch


def test_batch_writes_24_rows_with_version_and_timestamp(store, tmp_path, monkeypatch):
    out = tmp_path / "predictions" / "latest.csv"
    monkeypatch.setattr(config, "PREDICTIONS_URI", str(out))
    batch.main()

    df = pd.read_csv(out, parse_dates=["valid_time", "feature_time", "created_at"])
    assert list(df.columns) == [
        "valid_time", "hours_ahead", "temperature_c", "model_version",
        "feature_time", "feature_source", "created_at",
    ]  # fmt: skip
    assert len(df) == 24
    assert (df["valid_time"] - df["feature_time"] == pd.to_timedelta(df["hours_ahead"], "h")).all()
    assert set(df["model_version"].astype(str)) == {"1"}
    assert set(df["feature_source"]) == {"snapshot"}
