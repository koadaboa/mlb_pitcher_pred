import sqlite3
from pathlib import Path

import pandas as pd
import httpx

from src.scripts import fetch_strikeout_odds


class DummyResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def _mock_odds_data():
    return [
        {
            "id": "evt1",
            "commence_time": "2025-06-01T18:00:00Z",
            "bookmakers": [
                {
                    "title": "FakeBook",
                    "last_update": "2025-06-01T10:00:00Z",
                    "markets": [
                        {
                            "key": "player_strikeouts",
                            "outcomes": [
                                {"name": "John Doe Over 5.5", "price": 110, "point": 5.5},
                                {"name": "John Doe Under 5.5", "price": -130, "point": 5.5},
                            ],
                        }
                    ],
                }
            ],
        }
    ]


def test_fetch_and_save(monkeypatch, tmp_path: Path) -> None:
    def fake_get(self, url, params=None):
        return DummyResponse(_mock_odds_data())

    monkeypatch.setattr(httpx.Client, "get", fake_get)

    data = fetch_strikeout_odds.fetch_odds("key")
    df = fetch_strikeout_odds.odds_to_df(data)
    assert len(df) == 2

    db_path = tmp_path / "odds.db"
    fetch_strikeout_odds.save_odds(df, db_path)

    with sqlite3.connect(db_path) as conn:
        stored = pd.read_sql_query("SELECT * FROM strikeout_prop_odds", conn)
    assert len(stored) == 2
    assert set(stored["side"]) == {"over", "under"}
