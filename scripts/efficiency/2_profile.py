"""Step 2 · Profile before you optimise. Offline: committed snapshot, config of v1.

uv run --group efficiency python scripts/efficiency/2_profile.py

Runs train() twice under cProfile: once as it ships, once with build_features() swapped
for a version written the way it often is (a Python call per rolling window and per row).
Same features, same model; only where the time goes is different.
"""

import cProfile
import io
import pstats
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from _common import show, timed

from lucerne_temp.common.registry import FileModelStore
from lucerne_temp.features.build import build_features
from lucerne_temp.features.ingest import load_snapshot
from lucerne_temp.training.train import make_dataset, train

STAGES = {  # function name in the profile -> stage
    "load_snapshot": "load", "make_dataset": "features", "fit": "train",
    "evaluate": "evaluate", "save": "save",
}  # fmt: skip


def build_features_slow(temps: pd.Series, hours_ahead: int) -> pd.DataFrame:
    """Same columns as build_features(), written with Python calls per window and per row."""
    temps = temps.asfreq("h")
    X = pd.DataFrame({"temp_now": temps})
    for k in (1, 2, 3):
        X[f"lag_{k}"] = temps.shift(k)
    X["roll_mean_6"] = temps.rolling(6).apply(lambda w: w.mean())  # one call per window
    X["hour_sin"] = X.apply(lambda row: np.sin(2 * np.pi * row.name.hour / 24), axis=1)
    X["hour_cos"] = X.apply(lambda row: np.cos(2 * np.pi * row.name.hour / 24), axis=1)
    X["hours_ahead"] = hours_ahead
    X["same_hour_yesterday"] = temps.shift(24 - hours_ahead)
    return X


def profiled_train(store: FileModelStore) -> tuple[dict, pstats.Stats]:
    profiler = cProfile.Profile()
    profiler.enable()
    info = train("1", store)
    profiler.disable()
    return info, pstats.Stats(profiler)


def stage_times(stats: pstats.Stats) -> dict:
    """Cumulative seconds of the pipeline's own functions, read from the profile."""
    times = {}
    for (file, _, name), (_, _, _, cumulative, _) in stats.stats.items():
        if "lucerne_temp" in file and name in STAGES:
            times[STAGES[name]] = cumulative
    return times


def main() -> None:
    store = FileModelStore(artifact_dir=Path(tempfile.mkdtemp()))
    temps = load_snapshot()
    same = all(
        np.allclose(build_features(temps, h), build_features_slow(temps, h), equal_nan=True)
        for h in (1, 12, 24)
    )
    print(f"slow and fast build the same features: {same}")

    rows, slow_stats = [], None
    for label, builder in [("as shipped", build_features), ("slow features", build_features_slow)]:
        with patch("lucerne_temp.training.train.build_features", builder):
            wall, info = timed(lambda: train("1", store), repeats=3, warmup=1)
            features, _ = timed(lambda: make_dataset(temps), repeats=3)
            _, stats = profiled_train(store)
        times = stage_times(stats)
        total = stats.total_tt
        rows.append(
            {
                "version": label,
                "wall s": round(float(np.median(wall)) / 1000, 2),
                "of it features s": round(float(np.median(features)) / 1000, 2),
                "profiled s": round(total, 2),
                **{f"{s} %": round(100 * times[s] / total) for s in STAGES.values()},
                "MAE": info["metrics"]["mae"],
            }
        )
        if builder is build_features_slow:
            slow_stats = stats
    show(rows, "Without profiler (wall, median of 3) and with (share of the profiled run)")

    out = io.StringIO()
    slow_stats.stream = out
    slow_stats.strip_dirs().sort_stats("tottime").print_stats(8)
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    start = next(i for i, line in enumerate(lines) if line.lstrip().startswith("ncalls"))
    print("\nSlow version, top 8 functions by own time (tottime):")
    print("\n".join(lines[start:]))


if __name__ == "__main__":
    main()
