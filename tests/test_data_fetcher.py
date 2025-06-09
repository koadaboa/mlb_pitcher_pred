import argparse
from datetime import datetime
import sqlite3
import pandas as pd

from src.scripts import data_fetcher


def _setup_dummy_fetcher(monkeypatch, tmp_path, dummy_df):
    # Avoid writing to real locations
    monkeypatch.setattr(data_fetcher.pb.cache, "enable", lambda: None)
    monkeypatch.setattr(data_fetcher.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(data_fetcher, "ensure_dir", lambda p: p)

    class DummyCM:
        def __init__(self, *a, **k):
            pass
    monkeypatch.setattr(data_fetcher, "CheckpointManager", DummyCM)

    def fake_fetch(self, *a, **k):
        return dummy_df.copy()

    monkeypatch.setattr(data_fetcher.DataFetcher, "fetch_with_retries", fake_fetch)

    args = argparse.Namespace(date="2024-03-15", seasons=None, parallel=False, mlb_api=False, debug=False)
    fetcher = data_fetcher.DataFetcher(args)
    fetcher.db_path = tmp_path / "test.db"
    with sqlite3.connect(fetcher.db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS statcast_pitchers(game_date TEXT, pitcher INTEGER)")
        conn.execute("CREATE TABLE IF NOT EXISTS statcast_batters(game_date TEXT)")
    return fetcher


def test_pitcher_single_date_filters_game_type(monkeypatch, tmp_path):
    dummy_df = pd.DataFrame({
        "game_pk": [1, 2],
        "game_date": ["2024-03-15", "2024-03-15"],
        "pitcher": [100, 100],
        "batter": [10, 20],
        "pitch_number": [1, 1],
        "game_type": ["S", "R"],
    })

    fetcher = _setup_dummy_fetcher(monkeypatch, tmp_path, dummy_df)
    result = fetcher._fetch_pitcher_statcast_single_date(100, "Test", datetime(2024, 3, 15).date())
    assert set(result["game_type"]) == {"S", "R"}
    assert len(result) == 2


def test_batter_single_date_filters_game_type(monkeypatch, tmp_path):
    dummy_df = pd.DataFrame({
        "game_pk": [1, 2],
        "game_date": ["2024-03-15", "2024-03-15"],
        "pitcher": [100, 100],
        "batter": [11, 22],
        "game_type": ["S", "R"],
    })

    fetcher = _setup_dummy_fetcher(monkeypatch, tmp_path, dummy_df)
    result = fetcher._fetch_batter_statcast_single_date(datetime(2024, 3, 15).date())
    assert set(result["game_type"]) == {"S", "R"}
    assert len(result) == 2
