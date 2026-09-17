import pytest

from lucerne_temp.training import promote


def test_alias_resolves_to_a_version(store):
    assert store.version_of("champion") == "7"


def test_alias_cannot_point_to_an_unknown_version(store):
    with pytest.raises(KeyError):
        store.set_alias("champion", "99")


def test_versions_differ(store):
    assert store.version_config("7")["feature_set"] != store.version_config("8")["feature_set"]
    assert store.load("7").features != store.load("8").features


def test_training_is_deterministic_and_matches_the_registry(store, trained_artifacts):
    # if this fails after a snapshot refresh: retrain with --register (see README)
    for version, info in trained_artifacts[1].items():
        recorded = store.version_config(version)["metrics"]
        assert info["metrics"]["mae"] == pytest.approx(recorded["mae"], abs=0.01)
        assert info["metrics"]["mae_persistence"] == recorded["mae_persistence"]


def fake_mae(monkeypatch, maes: dict):
    monkeypatch.setattr(promote, "ensure_trained", lambda version, store: version)
    monkeypatch.setattr(promote, "evaluate", lambda version, window: {"mae": maes[version]})


def test_gate_promotes_a_better_candidate(store, monkeypatch):
    fake_mae(monkeypatch, {"7": 2.0, "8": 1.5})
    assert promote.gate("8", store) is True
    assert store.version_of("champion") == "8"


def test_gate_keeps_the_champion_when_the_candidate_is_worse(store, monkeypatch):
    fake_mae(monkeypatch, {"7": 1.5, "8": 2.0})
    assert promote.gate("8", store) is False
    assert store.version_of("champion") == "7"


def test_gate_on_the_real_models(store, capsys):
    # EXERCISE.md part C relies on this: with the committed snapshot, v8 passes the gate.
    # If it fails after a snapshot refresh, the exercise text must change, not this test.
    assert promote.gate("8", store) is True
    assert store.version_of("champion") == "8"
    out = capsys.readouterr().out
    assert "candidate v8" in out and "champion v7" in out
