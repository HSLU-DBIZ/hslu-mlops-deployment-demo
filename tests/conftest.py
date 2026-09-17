"""Tests run offline: the live Open-Meteo call is replaced, artifacts go to a temp dir."""

import shutil

import httpx
import pytest

from lucerne_temp.common import config
from lucerne_temp.common.registry import FileModelStore
from lucerne_temp.features import ingest
from lucerne_temp.training.train import train


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory):
    """Train v7 and v8 once per test session."""
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    store = FileModelStore(artifact_dir=artifact_dir)
    return artifact_dir, {v: train(v, store) for v in ("7", "8")}


@pytest.fixture
def store(trained_artifacts, tmp_path, monkeypatch):
    """A registry copy with champion = 7, so tests never touch registry/registry.json."""
    registry_path = tmp_path / "registry.json"
    shutil.copy(config.REGISTRY_PATH, registry_path)
    monkeypatch.setattr(config, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(config, "ARTIFACT_DIR", trained_artifacts[0])
    store = FileModelStore()
    store.set_alias("champion", "7")
    return store


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("tests are offline")

    monkeypatch.setattr(httpx, "get", refuse)
    ingest._cache.clear()
    yield
    ingest._cache.clear()
