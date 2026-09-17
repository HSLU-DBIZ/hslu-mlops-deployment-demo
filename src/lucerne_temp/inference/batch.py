"""Scheduled batch inference: python -m lucerne_temp.inference.batch

Writes the next 24 hourly forecasts to PREDICTIONS_URI (default predictions/latest.csv).
Every row carries model_version and created_at, so the file is its own prediction log.
"""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from lucerne_temp.common import config
from lucerne_temp.features.build import serving_features
from lucerne_temp.features.ingest import latest_temps
from lucerne_temp.inference.model import load_champion


def predict_next_24h() -> pd.DataFrame:
    version, model = load_champion()
    temps, source = latest_temps()
    rows = []
    for hours_ahead in range(1, 25):
        X, valid, ft = serving_features(temps, hours_ahead)
        rows.append({
            "valid_time": valid,
            "hours_ahead": hours_ahead,
            "temperature_c": round(float(model.predict(X)[0]), 2),
            "model_version": version,
            "feature_time": ft,
            "feature_source": source,
            "created_at": datetime.now(UTC).replace(microsecond=0),
        })  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    uri = config.PREDICTIONS_URI
    if "://" not in uri:  # local path; gs://bucket/latest.csv works once gcsfs is installed
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
    df = predict_next_24h()
    df.to_csv(uri, index=False)
    print(f"{len(df)} forecasts from model v{df.model_version[0]} -> {uri}")


if __name__ == "__main__":
    main()
