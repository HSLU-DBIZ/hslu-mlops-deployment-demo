"""Step 1 · Measure, then benchmark honestly. Offline: committed snapshot, config of v1.

uv run --group efficiency python scripts/efficiency/1_measure.py

Measures the training pipeline as it ships (train() from lucerne_temp.training.train):
wall time over repeated runs, peak memory, artifact size. Then benchmarks prediction:
first call after loading, warm single row (median and p95), and batches.
"""

import resource
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
from _common import show, timed
from threadpoolctl import threadpool_limits

from lucerne_temp.common.registry import FileModelStore
from lucerne_temp.features.build import serving_features
from lucerne_temp.features.ingest import load_snapshot
from lucerne_temp.training.train import TEST_DAYS, make_dataset, train

RUNS = 7
REPEATS = 200


def main() -> None:
    store = FileModelStore(artifact_dir=Path(tempfile.mkdtemp()))

    # --- measure: the whole training run, several times
    fits, _ = timed(lambda: train("1", store), repeats=RUNS)
    with threadpool_limits(limits=1):  # the tree model uses every core by default
        fits_1, _ = timed(lambda: train("1", store), repeats=3)
    tracemalloc.start()
    train("1", store)
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3  # bytes on macOS, KB on Linux
    show(
        [
            {"what": "train() wall time, first run", "value": f"{fits[0]:.0f} ms"},
            {
                "what": f"train() wall time, runs 2-{RUNS} median",
                "value": f"{np.median(fits[1:]):.0f} ms",
            },
            {
                "what": f"train() wall time, runs 2-{RUNS} min / max",
                "value": f"{fits[1:].min():.0f} / {fits[1:].max():.0f} ms",
            },
            {
                "what": "train() wall time, 1 thread, median of 3",
                "value": f"{np.median(fits_1):.0f} ms",
            },
            {"what": "peak Python memory during train()", "value": f"{peak_mb:.0f} MB"},
            {"what": "peak process memory (RSS) so far", "value": f"{rss_mb:.0f} MB"},
            {
                "what": "artifact size",
                "value": f"{store.artifact_path('1').stat().st_size / 1024:.0f} KB",
            },
        ],
        "Measure: train v1 on the snapshot",
    )

    # --- benchmark: prediction, as the API does it
    temps = load_snapshot()
    t0 = time.perf_counter()
    model = store.load("1")  # a fresh copy that has never predicted
    load_ms = (time.perf_counter() - t0) * 1000
    X, _, _ = serving_features(temps, hours_ahead=3)
    first, _ = timed(lambda: model.predict(X))
    row, _ = timed(lambda: model.predict(X), repeats=REPEATS, warmup=10)
    request, _ = timed(lambda: model.predict(serving_features(temps, 3)[0]), REPEATS, 10)
    with threadpool_limits(limits=1):
        row_1, _ = timed(lambda: model.predict(X), repeats=REPEATS, warmup=10)

    data = make_dataset(temps)
    test = data[data.index > data.index.max() - np.timedelta64(TEST_DAYS, "D")]
    one_time = data[data.index == test.index.min()]  # one feature time, 24 horizons
    loop24, _ = timed(lambda: [model.predict(one_time.iloc[[i]]) for i in range(24)], 30, 3)
    batch24, _ = timed(lambda: model.predict(one_time), repeats=REPEATS, warmup=10)
    batch_all, _ = timed(lambda: model.predict(test), repeats=30, warmup=3)

    def stats(ms: np.ndarray, rows: int = 1) -> dict:
        return {
            "median ms": round(float(np.median(ms)), 3),
            "p95 ms": round(float(np.percentile(ms, 95)), 3),
            "us per row": round(float(np.median(ms)) * 1000 / rows, 1),
        }

    show(
        [
            {"call": "joblib.load(artifact)", **stats(np.array([load_ms]))},
            {"call": "first predict, 1 row", **stats(first)},
            {"call": f"warm predict, 1 row (x{REPEATS})", **stats(row)},
            {"call": "warm predict, 1 row, 1 thread", **stats(row_1)},
            {"call": "features + predict, 1 row", **stats(request)},
            {"call": "24 horizons, 24 calls", **stats(loop24, 24)},
            {"call": "24 horizons, 1 call", **stats(batch24, 24)},
            {"call": f"test window, {len(test)} rows", **stats(batch_all, len(test))},
        ],
        "Benchmark: predict with the v1 artifact",
    )


if __name__ == "__main__":
    main()
