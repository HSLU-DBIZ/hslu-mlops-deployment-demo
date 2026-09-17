"""Raw data access: live Open-Meteo call for serving, committed snapshot for training."""

import json
import time

import httpx
import pandas as pd

from lucerne_temp.common import config


def parse_open_meteo(payload: dict) -> pd.Series:
    hourly = payload["hourly"]
    index = pd.to_datetime(hourly["time"], utc=True)
    temps = pd.Series(hourly["temperature_2m"], index=index, dtype=float, name="temperature_c")
    # the API also returns the rest of today as a forecast: keep observed hours only
    return temps[: pd.Timestamp.now(tz="UTC").floor("h")].dropna()


def fetch_live(past_days: int = 3) -> pd.Series:
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": "temperature_2m",
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": "UTC",
    }
    response = httpx.get(config.OPEN_METEO_URL, params=params, timeout=config.LIVE_TIMEOUT_S)
    response.raise_for_status()
    return parse_open_meteo(response.json())


def load_snapshot() -> pd.Series:
    df = pd.read_csv(config.SNAPSHOT_PATH, index_col="time", parse_dates=["time"])
    return df["temperature_c"]


_cache: dict = {}


def latest_temps() -> tuple[pd.Series, str]:
    """Newest raw hours and where they came from: "live" or "snapshot"."""
    if _cache and _cache["expires"] > time.monotonic():
        return _cache["value"]
    try:
        value = fetch_live(), "live"
    except Exception as exc:  # any failure of the live call must not take the service down
        print(json.dumps({"event": "live_fetch_failed", "error": repr(exc)}), flush=True)
        value = load_snapshot(), "snapshot"
    _cache.update(value=value, expires=time.monotonic() + config.LIVE_CACHE_S)
    return value
