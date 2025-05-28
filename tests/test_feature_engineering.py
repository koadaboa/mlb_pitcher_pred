import sqlite3
from pathlib import Path
import pandas as pd

from src.features import (
    engineer_pitcher_features,
    engineer_opponent_features,
    engineer_contextual_features,
    build_model_features,
)
from src.features.engineer_features import add_rolling_features
from src.features.contextual import _add_group_rolling


def setup_test_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        pitcher_df = pd.DataFrame(
            {
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
            }
        )
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
        assert any(col == "strikeouts_mean_3" for col in df.columns)
        assert all("_mean_5" not in c for c in df.columns)
        # ensure raw game stats are dropped
        assert "pitches" not in df.columns


def test_old_window_columns_removed(tmp_path: Path) -> None:
    """Ensure build_model_features drops stats from unsupported window sizes."""
    db_path = setup_test_db(tmp_path)

    engineer_pitcher_features(db_path=db_path)
    engineer_opponent_features(db_path=db_path)
    engineer_contextual_features(db_path=db_path)

    # Manually add a column using an old window size
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM rolling_pitcher_features", conn)
        df["strikeouts_mean_5"] = 1
        df.to_sql("rolling_pitcher_features", conn, if_exists="replace", index=False)

    build_model_features(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM model_features", conn)
        assert all("_mean_5" not in c for c in df.columns)


def test_rolling_stats_isolated_by_group() -> None:
    df = pd.DataFrame(
        {
            "game_pk": [1, 2, 3, 4],
            "game_date": pd.to_datetime([
                "2024-04-01",
                "2024-04-02",
                "2024-04-01",
                "2024-04-02",
            ]),
            "pitcher_id": [1, 1, 2, 2],
            "strikeouts": [5, 6, 7, 8],
        }
    )

    out = add_rolling_features(
        df,
        group_col="pitcher_id",
        date_col="game_date",
        windows=[2],
        numeric_cols=["strikeouts"],
    )

    assert pd.isna(out.loc[2, "strikeouts_mean_2"])
    assert out.loc[3, "strikeouts_mean_2"] == 7


def test_group_rolling_multiple_keys() -> None:
    df = pd.DataFrame(
        {
            "game_pk": [1, 2, 3, 4],
            "game_date": pd.to_datetime([
                "2024-04-01",
                "2024-04-08",
                "2024-04-01",
                "2024-04-08",
            ]),
            "pitcher_id": [10, 10, 20, 20],
            "opponent_team": ["A", "A", "B", "B"],
            "strikeouts": [5, 6, 7, 8],
        }
    )

    res = _add_group_rolling(
        df,
        ["pitcher_id", "opponent_team"],
        "game_date",
        prefix="opp_",
        windows=[2],
        n_jobs=1,
        numeric_cols=["strikeouts"],
    )

    assert pd.isna(res.loc[2, "opp_strikeouts_mean_2"])
    assert res.loc[3, "opp_strikeouts_mean_2"] == 7
