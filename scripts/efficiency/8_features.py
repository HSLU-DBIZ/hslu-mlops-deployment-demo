"""Step 8 · Incremental feature pipeline, and what the storage format costs.

uv run --group efficiency python scripts/efficiency/8_features.py

1. Incremental fetch: pretend the stored history ends a week early, run update() (fetch
   from watermark - overlap, merge, store) and compare with fetching everything again.
2. Incremental features: build features only for the new hours (plus the lookback they
   need) and check they equal a full rebuild.
3. The feature table of three years (make_dataset) as CSV and as Parquet, float64 and
   float32. Files go to data/history/demo/ (git-ignored).
"""

import os
import shutil
import time

import numpy as np
import pandas as pd
from _common import DAILY, fit_lightgbm, history_dataset, mae, show, split, timed

from lucerne_temp.features import history
from lucerne_temp.features.build import build_features
from lucerne_temp.training.folds import time_folds

DEMO = history.HISTORY_DIR / "demo"
LOOKBACK_HOURS = 24  # the oldest value build_features() reads: 24 h before the valid time


def incremental_fetch() -> pd.Series:
    full = history.load_history()
    path, raw_dir = DEMO / "temperature.parquet", DEMO / "raw"
    history.write_table(full[: full.index.max() - pd.Timedelta(days=7)], path)
    fetch = lambda start, end: history.fetch_archive_range(start, end, raw_dir)  # noqa: E731
    t0 = time.perf_counter()
    merged, fetched = history.update(path, history.END, fetch)
    update_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    refetched = fetch(history.START, history.END)
    full_s = time.perf_counter() - t0
    raw_kb = {p.name: p.stat().st_size // 1024 for p in raw_dir.glob("*.json")}
    full_kb = raw_kb[f"{history.START}_{history.END}.json"]
    show(
        [
            {
                "run": f"update(): watermark - {history.OVERLAP_HOURS} h to end, merge, store",
                "hours fetched": fetched,
                "response KB": sum(raw_kb.values()) - full_kb,
                "s": round(update_s, 2),
            },
            {
                "run": "full refetch: 3 years, fetch only",
                "hours fetched": len(refetched),
                "response KB": full_kb,
                "s": round(full_s, 2),
            },
        ],
        f"Fetch (stored table ended {full.index.max() - pd.Timedelta(days=7)})",
    )
    print(f"merged table equals the full history: {merged.equals(full)}")
    print(
        f"raw responses kept in {raw_dir.relative_to(history.HISTORY_DIR.parents[1])}: "
        + ", ".join(sorted(raw_kb))
    )
    return full


def incremental_features(temps: pd.Series) -> None:
    new_from = temps.index.max() - pd.Timedelta(days=7)
    tail = temps[temps.index > new_from - pd.Timedelta(hours=LOOKBACK_HOURS)]
    rows = []
    for label, source in [("full rebuild, 3 years", temps), ("new week + 24 h lookback", tail)]:
        ms, feats = timed(lambda: [build_features(source, h) for h in range(1, 25)], 5, 1)  # noqa: B023
        rows.append(
            {
                "features for": label,
                "input hours": len(source),
                "median ms (24 horizons)": round(float(np.median(ms)), 1),
            }
        )
        if source is tail:
            same = all(
                np.allclose(
                    build_features(temps, h).loc[f.index[f.index > new_from]],
                    f[f.index > new_from],
                    equal_nan=True,
                )
                for h, f in zip(range(1, 25), feats, strict=True)
            )
    show(rows, "Features")
    print(f"new rows identical to the full rebuild: {same}")


def formats() -> None:
    data = history_dataset()
    numeric = data.drop(columns="valid_time")
    csv, parquet = DEMO / "features.csv", DEMO / "features.parquet"
    write_csv, _ = timed(lambda: data.to_csv(csv))
    write_pq, _ = timed(lambda: data.to_parquet(parquet))
    read_csv, _ = timed(lambda: pd.read_csv(csv, index_col=0, parse_dates=[0, "valid_time"]), 3, 1)
    read_pq, _ = timed(lambda: pd.read_parquet(parquet), 3, 1)
    read_2, _ = timed(lambda: pd.read_parquet(parquet, columns=["temp_now", "target"]), 3, 1)
    mb = lambda path: round(os.path.getsize(path) / 1e6, 1)  # noqa: E731
    csv_mb, parquet_mb = mb(csv), mb(parquet)
    csv.unlink()  # nothing reads it again
    show(
        [
            {
                "format": "CSV",
                "MB on disk": csv_mb,
                "write s": round(write_csv[0] / 1000, 2),
                "read s (median of 3)": round(float(np.median(read_csv)) / 1000, 3),
            },
            {
                "format": "Parquet",
                "MB on disk": parquet_mb,
                "write s": round(write_pq[0] / 1000, 2),
                "read s (median of 3)": round(float(np.median(read_pq)) / 1000, 3),
            },
            {
                "format": "Parquet, 2 columns",
                "MB on disk": "",
                "read s (median of 3)": round(float(np.median(read_2)) / 1000, 3),
            },
        ],
        f"Feature table: {len(data)} rows x {data.shape[1]} columns",
    )

    train, test = split(data)
    ((a, b),) = time_folds(train, n_splits=1, test_days=90)
    rows = []
    for dtype in ("float64", "float32"):
        cast = lambda d: d.astype({c: dtype for c in numeric.columns})  # noqa: B023, E731
        fit, val, scored = cast(train.iloc[a]), cast(train.iloc[b]), cast(test)
        model = fit_lightgbm(fit, val, DAILY)
        rows.append(
            {
                "dtype": dtype,
                "memory MB": round(cast(data).memory_usage(deep=True).sum() / 1e6),
                "LightGBM test MAE": round(mae(scored["target"], model.predict(scored)), 3),
            }
        )
    show(rows, "Same table in memory")


def main() -> None:
    shutil.rmtree(DEMO, ignore_errors=True)
    temps = incremental_fetch()
    incremental_features(temps)
    formats()


if __name__ == "__main__":
    main()
