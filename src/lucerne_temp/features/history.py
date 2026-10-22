"""Three years of hourly history for the efficiency walkthrough (EFFICIENCY.md).

Source: the Open-Meteo archive API (reanalysis, no key). load_history() fetches the fixed
period once and caches it as Parquet in data/history/ (git-ignored); later calls read the
cache. update() is the incremental version: it fetches only from the watermark (the newest
hour already stored) minus an overlap, and merges. Every raw response is kept next to the
table, so a parsing bug can be fixed without fetching again.

The service, the training of v1 and v2 and the tests do not use this: they read the
committed snapshot. Parquet needs pyarrow: uv run --group efficiency ...
"""

from collections.abc import Callable
from pathlib import Path

import httpx
import pandas as pd

from lucerne_temp.common import config
from lucerne_temp.features.ingest import parse_open_meteo

# fixed, so everybody gets the same numbers; archive data lag a few days behind today
START, END = "2023-10-01", "2026-09-30"
HISTORY_DIR = config.ROOT / "data" / "history"
TABLE_PATH = HISTORY_DIR / "temperature.parquet"
RAW_DIR = HISTORY_DIR / "raw"
OVERLAP_HOURS = 48  # refetch the newest stored hours too: the archive corrects recent days


def fetch_archive_range(start: str, end: str, raw_dir: Path = RAW_DIR) -> pd.Series:
    """Hourly temperature from start to end (dates, inclusive, UTC). Keeps the raw JSON."""
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "timezone": "UTC",
    }
    response = httpx.get(config.OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{start}_{end}.json").write_text(response.text)
    return parse_open_meteo(response.json())


def merge(old: pd.Series, new: pd.Series) -> pd.Series:
    """Append new hours. Where both have an hour, the newer fetch wins (late corrections)."""
    combined = pd.concat([old, new])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def check_complete(temps: pd.Series) -> None:
    missing = temps.asfreq("h").isna().sum()
    if missing:
        raise ValueError(f"history has {missing} missing hours")


def read_table(path: Path = TABLE_PATH) -> pd.Series:
    return pd.read_parquet(path)["temperature_c"]


def write_table(temps: pd.Series, path: Path = TABLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temps.rename("temperature_c").rename_axis("time").to_frame().to_parquet(path)


def update(
    path: Path,
    end: str,
    fetch: Callable[[str, str], pd.Series] = fetch_archive_range,
    overlap_hours: int = OVERLAP_HOURS,
) -> tuple[pd.Series, int]:
    """Incremental run: fetch (watermark - overlap) to end, merge, store.

    Returns the merged series and how many hours were fetched.
    """
    old = read_table(path)
    watermark = old.index.max()
    start = watermark - pd.Timedelta(hours=overlap_hours)
    new = fetch(f"{start:%Y-%m-%d}", end)
    merged = merge(old, new)
    check_complete(merged)
    write_table(merged, path)
    return merged, len(new)


def load_history(path: Path = TABLE_PATH) -> pd.Series:
    """The fixed three-year period; fetched on first use (one call, about 1 s)."""
    if not path.exists():
        temps = fetch_archive_range(START, END)
        check_complete(temps)
        write_table(temps, path)
        where = path.relative_to(config.ROOT) if path.is_relative_to(config.ROOT) else path
        print(f"fetched {len(temps)} hours, {temps.index[0]} to {temps.index[-1]} -> {where}")
    return read_table(path)


if __name__ == "__main__":  # uv run --group efficiency python -m lucerne_temp.features.history
    temps = load_history()
    size_kb = TABLE_PATH.stat().st_size // 1024
    print(
        f"{len(temps)} hours, {temps.index[0]} to {temps.index[-1]}, {TABLE_PATH.name} {size_kb} KB"
    )
