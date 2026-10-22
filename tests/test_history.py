"""The incremental history update, offline: fetch is a fake, the table lives in memory."""

import pandas as pd
import pytest

from lucerne_temp.features import history


def hours(start: str, n: int, value: float = 10.0) -> pd.Series:
    index = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.Series(value, index=index, name="temperature_c")


def test_merge_appends_and_newer_fetch_wins():
    old = hours("2026-09-01", 48, value=1.0)
    new = hours("2026-09-02", 48, value=2.0)  # overlaps the last 24 h of old
    merged = history.merge(old, new)
    assert len(merged) == 72
    assert merged.index.is_monotonic_increasing
    assert (merged[:"2026-09-01 23:00"] == 1.0).all()
    assert (merged["2026-09-02":] == 2.0).all()  # corrected values replace the stored ones


def test_check_complete_finds_a_missing_hour():
    temps = hours("2026-09-01", 48)
    history.check_complete(temps)
    with pytest.raises(ValueError, match="1 missing hours"):
        history.check_complete(temps.drop(temps.index[10]))


def test_update_fetches_from_watermark_minus_overlap(monkeypatch):
    table = {"t": hours("2026-09-01", 24 * 10)}  # ends 2026-09-10 23:00
    monkeypatch.setattr(history, "read_table", lambda path: table["t"])
    monkeypatch.setattr(history, "write_table", lambda temps, path: table.update(t=temps))
    calls = []

    def fake_fetch(start: str, end: str) -> pd.Series:
        calls.append((start, end))
        return hours(start, 24 * 5, value=3.0)  # start .. start + 5 days

    merged, fetched = history.update("unused", "2026-09-12", fake_fetch, overlap_hours=48)
    assert calls == [("2026-09-08", "2026-09-12")]  # watermark 09-10 23:00 minus 48 h
    assert fetched == 120
    assert merged.index.max() == pd.Timestamp("2026-09-12 23:00", tz="UTC")
    assert table["t"] is merged


def test_table_round_trip(tmp_path):
    pytest.importorskip("pyarrow")  # installed by: uv sync --group efficiency
    temps = hours("2026-09-01", 24)
    history.write_table(temps, tmp_path / "t.parquet")
    assert history.read_table(tmp_path / "t.parquet").equals(temps)
