"""Fetch the training snapshot once and commit it: python -m lucerne_temp.features.backfill

Default source is the forecast API with past_days, the same source the service reads at
serving time. Its history is complete for about 70 days. For a longer history use
--source archive (reanalysis data; differs from the forecast API by about 0.9 °C on average).
"""

import argparse

import httpx
import pandas as pd

from lucerne_temp.common import config
from lucerne_temp.features.ingest import fetch_live, parse_open_meteo


def fetch_archive(days: int) -> pd.Series:
    today = pd.Timestamp.now(tz="UTC").normalize()
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": f"{today - pd.Timedelta(days=days):%Y-%m-%d}",
        "end_date": f"{today:%Y-%m-%d}",
        "hourly": "temperature_2m",
        "timezone": "UTC",
    }
    response = httpx.get(config.OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    return parse_open_meteo(response.json())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=70)
    parser.add_argument("--source", choices=["forecast", "archive"], default="forecast")
    args = parser.parse_args()

    config.LIVE_TIMEOUT_S = 30
    temps = fetch_live(args.days) if args.source == "forecast" else fetch_archive(args.days)
    assert temps.asfreq("h").notna().all(), "snapshot has gaps"
    config.SNAPSHOT_PATH.parent.mkdir(exist_ok=True)
    temps.rename_axis("time").to_csv(config.SNAPSHOT_PATH)
    print(f"{len(temps)} hours, {temps.index[0]} to {temps.index[-1]} -> {config.SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
