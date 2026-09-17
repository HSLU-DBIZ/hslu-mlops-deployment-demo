"""Resolve the alias champion to a version, then load that version."""

from lucerne_temp.common.config import ALIAS
from lucerne_temp.common.model import Forecaster
from lucerne_temp.common.registry import FileModelStore
from lucerne_temp.training.train import ensure_trained


def load_champion() -> tuple[str, Forecaster]:
    store = FileModelStore()
    version = store.version_of(ALIAS)  # the version goes into /health and every prediction
    return version, ensure_trained(version, store)
