import sqlite3
from pathlib import Path
import pandas as pd

from src.features import (
    engineer_pitcher_features,
    engineer_opponent_features,
    engineer_contextual_features,
    build_model_features,
)


def setup_test_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        pitcher_df = pd.DataFrame({
            "game_pk": [1, 2, 3],
            "game_date": pd.to_datetime(["2024-04-01", "2024-04-08", "2024-04-15"]),
            "pitcher_id": [10, 10, 10],
            "opponent_team": ["A", "B", "C"],
            "home_team": ["H1", "H1", "H2"],
            "hp_umpire": ["U1", "U1", "U2"],
            "weather": ["Sunny", "Cloudy", "Sunny"],
            "temp": [70, 65, 60],
            "wind": ["5 mph", "10 mph", "5 mph"],
            "elevation": [500, 500, 600],
            "strikeouts": [5, 6, 7],
            "pitches": [80, 85, 90],
        })
        matchup_df = pitcher_df.copy()
        pitcher_df.to_sql("game_level_starting_pitchers", conn, index=False)
        matchup_df.to_sql("game_level_matchup_details", conn, index=False)
    return db_path


def test_feature_pipeline(tmp_path: Path) -> None:
    db_path = setup_test_db(tmp_path)

    engineer_pitcher_features(db_path=db_path)
    engineer_opponent_features(db_path=db_path)
    engineer_contextual_features(db_path=db_path)
    build_model_features(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='model_features'"
        )
        assert cur.fetchone() is not None
        df = pd.read_sql_query("SELECT * FROM model_features", conn)
        assert len(df) == 3
        assert any(col.startswith("strikeouts_mean_") for col in df.columns)
        assert not any(col.endswith("_x") or col.endswith("_y") for col in df.columns)
