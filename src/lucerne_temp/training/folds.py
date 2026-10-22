"""Time-ordered cross-validation folds for make_dataset() rows.

Every feature time t has one row per horizon, so rows are not in time order and a row's
target lies hours_ahead after its index. Random KFold mixes the same hours into train and
test. These folds cut by time instead: the test block is (cut, cut + test_days], and a
training row is used only if its target was observed by the cut (valid_time <= cut). The
gap therefore equals each row's horizon, as in train.py's out-of-time split.
"""

import numpy as np
import pandas as pd


def time_folds(
    data: pd.DataFrame, n_splits: int, test_days: int, gap: bool = True
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) positions, expanding window, last fold ends at the newest row.

    gap=False keeps training rows whose target lies after the cut: the leaky version,
    there only to measure what the gap is worth.
    """
    end = data.index.max()
    block = pd.Timedelta(days=test_days)
    folds = []
    for k in range(n_splits, 0, -1):
        cut = end - k * block
        seen = data["valid_time"] if gap else data.index
        train = np.flatnonzero(seen <= cut)
        test = np.flatnonzero((data.index > cut) & (data.index <= cut + block))
        folds.append((train, test))
    return folds
