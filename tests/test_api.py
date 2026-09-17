import json

import pytest
from fastapi.testclient import TestClient

from lucerne_temp.features.ingest import load_snapshot
from lucerne_temp.inference.api import app


@pytest.fixture
def client(store):
    with TestClient(app) as client:  # the with block runs lifespan: the champion is loaded
        yield client


def test_health_reports_model_version_and_feature_time(client):
    body = client.get("/health").json()
    assert set(body) == {"status", "model_version", "feature_time", "feature_source"}
    assert body["status"] == "ok"
    assert body["model_version"] == "7"


def test_predict_response_fields(client):
    response = client.post("/predict", json={"hours_ahead": 3})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "valid_time", "temperature_c", "model_version", "feature_time", "feature_source",
    }  # fmt: skip
    assert body["model_version"] == "7"
    assert -30 < body["temperature_c"] < 45


def test_predict_defaults_to_one_hour(client):
    assert client.post("/predict", json={}).status_code == 200


@pytest.mark.parametrize("bad", [{"hours_ahead": 0}, {"hours_ahead": 25}, {"hours_ahead": "soon"}])
def test_bad_input_is_a_422(client, bad):
    assert client.post("/predict", json=bad).status_code == 422


def test_falls_back_to_snapshot_when_live_call_fails(client):
    # conftest makes every httpx.get fail, as if Open-Meteo were down
    body = client.post("/predict", json={"hours_ahead": 1}).json()
    snapshot_end = load_snapshot().index[-1]
    assert body["feature_source"] == "snapshot"
    assert body["feature_time"] == snapshot_end.isoformat().replace("+00:00", "Z")


def test_live_features_are_used_when_the_call_succeeds(client, monkeypatch):
    from lucerne_temp.features import ingest

    monkeypatch.setattr(ingest, "fetch_live", lambda: load_snapshot().iloc[:-5])
    body = client.get("/health").json()
    assert body["feature_source"] == "live"


def test_every_prediction_is_logged_as_one_json_line(client, capsys):
    client.post("/predict", json={"hours_ahead": 2})
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    log = next(line for line in lines if line["event"] == "prediction")
    assert {"ts", "model_version", "feature_time", "features", "hours_ahead",
            "temperature_c", "latency_ms"} <= set(log)  # fmt: skip


@pytest.mark.parametrize("version, logs_daily_feature", [("7", False), ("8", True)])
def test_log_holds_exactly_the_features_the_loaded_model_uses(
    store, capsys, version, logs_daily_feature
):
    store.set_alias("champion", version)
    with TestClient(app) as client:
        client.post("/predict", json={"hours_ahead": 2})
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    log = next(line for line in lines if line["event"] == "prediction")
    assert list(log["features"]) == store.load(version).features
    assert ("same_hour_yesterday" in log["features"]) is logs_daily_feature


def test_service_follows_the_alias(store):
    store.set_alias("champion", "8")
    with TestClient(app) as client:
        assert client.get("/health").json()["model_version"] == "8"
