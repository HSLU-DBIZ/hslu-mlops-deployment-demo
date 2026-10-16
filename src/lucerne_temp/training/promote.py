"""Promotion gate: python scripts/promote.py 2

Scores candidate and champion on the same recent window with the same metric and moves
the alias champion only if the candidate is better. Exit code 1 means "not promoted".
"""

import argparse
import sys

import pandas as pd

from lucerne_temp.common.config import ALIAS
from lucerne_temp.common.registry import FileModelStore
from lucerne_temp.features.ingest import load_snapshot
from lucerne_temp.training.train import ensure_trained, evaluate, make_dataset

WINDOW_DAYS = 7  # inside the out-of-time test window: neither model was trained on it


def gate(candidate: str, store: FileModelStore | None = None) -> bool:
    store = store or FileModelStore()
    champion = store.version_of(ALIAS)
    data = make_dataset(load_snapshot())
    window = data[data.index > data.index.max() - pd.Timedelta(days=WINDOW_DAYS)]

    # same window, same metric
    new = evaluate(ensure_trained(candidate, store), window)["mae"]
    old = evaluate(ensure_trained(champion, store), window)["mae"]
    print(f"last {WINDOW_DAYS} days: candidate v{candidate} MAE {new:.3f} °C, "
          f"champion v{champion} MAE {old:.3f} °C")  # fmt: skip
    if new < old:
        store.set_alias(ALIAS, candidate)
        print(f"promoted: {ALIAS} -> v{candidate}. Commit registry/registry.json and deploy.")
        return True
    print(f"not promoted: {ALIAS} stays v{champion}. Nothing to deploy.")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="candidate version, e.g. 2")
    parser.add_argument("--force", action="store_true",
                        help="move the alias without the gate (model rollback)")  # fmt: skip
    args = parser.parse_args()
    if args.force:
        FileModelStore().set_alias(ALIAS, args.version)
        print(f"{ALIAS} -> v{args.version} (forced, gate skipped)")
        return
    sys.exit(0 if gate(args.version) else 1)


if __name__ == "__main__":
    main()
