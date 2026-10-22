import pandas as pd
import pytest

from lucerne_temp.features.ingest import load_snapshot
from lucerne_temp.training.folds import time_folds
from lucerne_temp.training.train import make_dataset


@pytest.fixture(scope="module")
def data():
    return make_dataset(load_snapshot())


def test_test_blocks_are_ordered_disjoint_and_end_at_the_newest_row(data):
    folds = time_folds(data, n_splits=3, test_days=7)
    assert len(folds) == 3
    blocks = [data.iloc[test].index for _, test in folds]
    for earlier, later in zip(blocks, blocks[1:], strict=False):
        assert earlier.max() < later.min()
    assert blocks[-1].max() == data.index.max()
    assert blocks[-1].max() - blocks[-1].min() < pd.Timedelta(days=7)


def test_no_training_target_lies_after_the_cut(data):
    for train, test in time_folds(data, n_splits=3, test_days=7):
        cut = data.iloc[test].index.min() - pd.Timedelta(hours=1)
        assert data.iloc[train]["valid_time"].max() <= cut
        assert len(set(train) & set(test)) == 0


def test_without_gap_targets_leak_into_the_test_block(data):
    train, test = time_folds(data, n_splits=1, test_days=7, gap=False)[0]
    assert data.iloc[train]["valid_time"].max() > data.iloc[test].index.min()
