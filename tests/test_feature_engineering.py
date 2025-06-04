import sqlite3
from pathlib import Path
import pandas as pd
import runpy
import sys

from src.features import (
    engineer_pitcher_features,
    engineer_opponent_features,
    engineer_contextual_features,
    engineer_lineup_trends,
    engineer_batter_pitcher_history,
    build_model_features,
)
from src.features.engineer_features import add_rolling_features
from src.config import StrikeoutModelConfig


def setup_test_db(tmp_path: Path, cross_season: bool = False) -> Path:
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        if cross_season:
            dates = ["2023-09-28", "2024-04-01", "2024-04-08"]
        else:
            dates = ["2024-04-01", "2024-04-08", "2024-04-15"]
        pitcher_df = pd.DataFrame(
            {
                "game_pk": [1, 2, 3],
                "game_date": pd.to_datetime(dates),
                "pitcher_id": [10, 10, 10],
                "pitcher_hand": ["R", "L", "R"],
                "opponent_team": ["A", "B", "C"],
                "home_team": ["ARI", "ARI", "BOS"],
                "hp_umpire": ["U1", "U1", "U2"],
                "weather": ["Sunny", "Cloudy", "Sunny"],
                "temp": [70, 65, 60],
                "humidity": [40, 50, 55],
                "wind": ["5 mph", "10 mph", "5 mph"],
                "elevation": [500, 500, 600],
                "strikeouts": [5, 6, 7],
                "pitches": [80, 85, 90],
                "fip": [4.0, 3.5, 3.0],
                "slider_pct": [0.2, 0.25, 0.3],
                "offspeed_to_fastball_ratio": [0.5, 0.6, 0.55],
                "fastball_then_breaking_rate": [0.3, 0.4, 0.35],
                "zone_pct": [0.5, 0.55, 0.6],
                "hard_hit_rate": [0.3, 0.25, 0.2],
                "unique_pitch_types": [3, 4, 3],
                "zone_pct": [0.5, 0.55, 0.6],
                "chase_rate": [0.2, 0.25, 0.3],
                "avg_launch_speed": [89, 90, 91],
                "max_launch_speed": [99, 100, 101],
                "avg_launch_angle": [10, 12, 15],
                "max_launch_angle": [25, 30, 35],
                "hard_hit_rate": [0.4, 0.45, 0.5],
                "barrel_rate": [0.1, 0.12, 0.15],
                "pfx_x": [0.1, 0.2, 0.15],
                "pfx_z": [-0.5, -0.6, -0.4],
                "release_extension": [6.0, 6.1, 6.2],
                "plate_x": [-0.2, -0.3, -0.1],
                "plate_z": [2.5, 2.6, 2.7],
            }
        )
        matchup_df = pitcher_df.copy()
        matchup_df["bat_avg"] = [0.25, 0.26, 0.27]
        matchup_df["bat_obp"] = [0.32, 0.33, 0.34]
        matchup_df["bat_slugging"] = [0.40, 0.41, 0.42]
        matchup_df["bat_ops"] = [0.72, 0.74, 0.76]
        matchup_df["bat_woba"] = [0.31, 0.32, 0.33]
        pitcher_df.to_sql("game_level_starting_pitchers", conn, index=False)
        matchup_df.to_sql("game_level_matchup_details", conn, index=False)

        batter_df = pd.DataFrame(
            {
                "game_pk": [1, 2, 3],
                "pitcher_id": [10, 10, 10],
                "batter_id": ["101", "102", "103"],
                "opponent_team": ["A", "B", "C"],
                "stand": ["R", "L", "L"],
                "plate_appearances": [4, 4, 4],
                "strikeouts": [1, 2, 1],
                "ops": [0.7, 0.75, 0.72],
                "swings": [10, 12, 11],
                "whiffs": [3, 4, 3],
            }
        )
        batter_df.to_sql("game_level_batters_vs_starters", conn, index=False)

        lineup_df = pd.DataFrame(
            {
                "game_pk": [1, 2, 3],
                "team": ["A", "B", "C"],
                "batter_id": ["101", "102", "103"],
                "lineup_avg_ops": [0.72, 0.73, 0.74],
                "projected_k_pct": [0.2, 0.22, 0.21],
            }
        )
        lineup_df.to_sql("game_starting_lineups", conn, index=False)

        players_df = pd.DataFrame(
            {"player_id": [10], "birth_date": pd.to_datetime(["1990-01-01"])}
        )
        players_df.to_sql("players", conn, index=False)

        catcher_df = pd.DataFrame(
            {
                "game_pk": [1, 2, 3],
                "catcher_id": [200, 200, 200],
                "game_date": pd.to_datetime(dates),
                "called_strike_rate": [0.5, 0.55, 0.6],
                "framing_runs": [1.0, 1.2, 1.4],
            }
        )
        catcher_df.to_sql("catcher_defense_metrics", conn, index=False)
    return db_path


def test_feature_pipeline(tmp_path: Path) -> None:
    db_path = setup_test_db(tmp_path)

    engineer_pitcher_features(db_path=db_path)
    engineer_opponent_features(db_path=db_path)
    engineer_contextual_features(db_path=db_path)
    engineer_lineup_trends(db_path=db_path)
    engineer_batter_pitcher_history(db_path=db_path)
    engineer_batter_pitcher_history(db_path=db_path)
    build_model_features(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='model_features'"
        )
        assert cur.fetchone() is not None
        df = pd.read_sql_query("SELECT * FROM model_features", conn)
        assert len(df) == 3
        assert any(col == "strikeouts_mean_3" for col in df.columns)
        assert any(col == "fip_mean_3" for col in df.columns)
        assert "slider_pct_mean_3" in df.columns
        assert "offspeed_to_fastball_ratio_mean_3" in df.columns
        assert "fastball_then_breaking_rate_mean_3" in df.columns
        assert "two_strike_k_rate_mean_3" in df.columns
        assert "high_leverage_k_rate_mean_3" in df.columns
        assert "woba_runners_on_mean_3" in df.columns
        assert "unique_pitch_types_mean_3" in df.columns
        assert "zone_pct_mean_3" in df.columns
        assert "hard_hit_rate_mean_3" in df.columns
        assert "pfx_x_mean_3" in df.columns
        assert "lineup_avg_ops_mean_3" in df.columns
        assert "team_k_rate_mean_3" in df.columns
        assert "opp_lineup_woba_mean_3" in df.columns
        assert "opp_lineup_pct_left_mean_3" in df.columns
        assert "strikeouts_mean_20" in df.columns
        assert "fip_mean_100" in df.columns
        halflife = StrikeoutModelConfig.EWM_HALFLIFE
        assert f"strikeouts_ewm_{halflife}" in df.columns
        assert f"strikeouts_momentum_ewm_{halflife}" in df.columns
        assert all("_mean_77" not in c for c in df.columns)
        # ensure raw game stats are dropped
        assert "pitches" not in df.columns
        assert "fip" not in df.columns
        assert "slider_pct" not in df.columns
        assert "team_k_rate" not in df.columns
        assert "park_factor" not in df.columns
        assert "venue_humidity_mean_3" in df.columns
        assert "venue_park_factor_mean_3" in df.columns
        # encoded categorical columns should be numeric
        assert "home_team_enc" in df.columns
        assert pd.api.types.is_numeric_dtype(df["home_team_enc"])
        assert "day_of_week" in df.columns
        assert "travel_distance" in df.columns
        assert "on_il" in df.columns
        assert "days_since_il" in df.columns
        assert "pitches_last_7d" in df.columns
        assert "season_ip_last_30d" in df.columns
        assert "pitcher_age" in df.columns
        assert "years_in_MLB" in df.columns
        assert "career_k_per9" in df.columns
        assert "career_fip" in df.columns
        assert pd.api.types.is_numeric_dtype(df["on_il"])
        # ensure merge suffixes were resolved
        assert "game_date" in df.columns
        assert not any(c.endswith("_x") or c.endswith("_y") for c in df.columns)
        assert "slot1_lineup_ops_mean_3" in df.columns
        assert "opp_batter_batter_so_rate_mean_3" in df.columns
        assert "projected_lineup_k_pct" in df.columns


def test_old_window_columns_removed(tmp_path: Path) -> None:
    """Ensure build_model_features drops stats from unsupported window sizes."""
    db_path = setup_test_db(tmp_path)

    engineer_pitcher_features(db_path=db_path)
    engineer_opponent_features(db_path=db_path)
    engineer_contextual_features(db_path=db_path)
    engineer_lineup_trends(db_path=db_path)

    # Manually add a column using an unsupported window size
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM rolling_pitcher_features", conn)
        df["strikeouts_mean_77"] = 1
        df["fip_mean_77"] = 1
        df["slider_pct_mean_77"] = 1
        df.to_sql("rolling_pitcher_features", conn, if_exists="replace", index=False)

    build_model_features(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM model_features", conn)
        assert all("_mean_77" not in c for c in df.columns)


def test_group_specific_rolling() -> None:
    df = pd.DataFrame(
        {
            "game_pk": [1, 2, 3, 4],
            "game_date": pd.to_datetime(
                ["2024-04-01", "2024-04-08", "2024-04-01", "2024-04-08"]
            ),
            "pitcher_id": [10, 10, 20, 20],
            "strikeouts": [5, 6, 7, 8],
            "pitches": [80, 90, 100, 110],
        }
    )
    result = add_rolling_features(
        df,
        group_col="pitcher_id",
        date_col="game_date",
        windows=[3],
        numeric_cols=["strikeouts"],
        ewm_halflife=StrikeoutModelConfig.EWM_HALFLIFE,
    )
    # First row for pitcher 20 should not include pitcher 10 data
    assert (
        pd.isna(result.loc[2, "strikeouts_mean_3"])
        or result.loc[2, "strikeouts_mean_3"] == 0
    )
    assert f"strikeouts_ewm_{StrikeoutModelConfig.EWM_HALFLIFE}" in result.columns


def test_log_features_added(tmp_path: Path) -> None:
    db_path = setup_test_db(tmp_path)

    engineer_pitcher_features(db_path=db_path)
    engineer_opponent_features(db_path=db_path)
    engineer_contextual_features(db_path=db_path)
    engineer_lineup_trends(db_path=db_path)
    engineer_batter_pitcher_history(db_path=db_path)
    build_model_features(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM model_features", conn)
        assert any(c.startswith("log_") for c in df.columns)


def test_split_metrics_rolled(tmp_path: Path) -> None:
    db_path = setup_test_db(tmp_path)

    engineer_pitcher_features(db_path=db_path)
    engineer_opponent_features(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM rolling_pitcher_vs_team", conn)
        assert "team_hand_bat_ops_vs_RHP_mean_3" in df.columns
        assert "team_hand_bat_k_rate_vs_LHP_mean_3" in df.columns


def test_base_context_fields_kept(tmp_path: Path) -> None:
    """Ensure contextual stats like park_factor are retained and transformed."""
    db_path = setup_test_db(tmp_path)

    engineer_pitcher_features(db_path=db_path)
    engineer_opponent_features(db_path=db_path)
    engineer_contextual_features(db_path=db_path)
    engineer_lineup_trends(db_path=db_path)
    engineer_batter_pitcher_history(db_path=db_path)
    build_model_features(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM model_features", conn)
        assert "park_factor" in df.columns
        assert "log_park_factor" in df.columns
        assert "day_of_week" in df.columns
        assert "travel_distance" in df.columns


def test_rest_days_across_seasons(tmp_path: Path) -> None:
    db_path = setup_test_db(tmp_path, cross_season=True)

    engineer_pitcher_features(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM rolling_pitcher_features", conn)
        assert "rest_days" in df.columns
        # First start has no prior appearance
        assert pd.isna(df.loc[0, "rest_days"])
        # Cross-season gap should be calculated correctly
        assert df.loc[1, "rest_days"] == 186
        assert df.loc[2, "rest_days"] == 7


def test_engineer_lineup_trends(tmp_path: Path) -> None:
    db_path = setup_test_db(tmp_path)

    engineer_lineup_trends(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM lineup_trends", conn)
        assert "lineup_avg_ops_mean_3" in df.columns
        assert "projected_lineup_k_pct" in df.columns


def test_run_feature_engineering_script(tmp_path: Path) -> None:
    """Ensure the CLI pipeline creates lineup and model feature tables."""
    db_path = setup_test_db(tmp_path)

    argv = ["run_feature_engineering", "--db-path", str(db_path)]
    orig_argv = sys.argv[:]
    sys.argv = argv
    try:
        runpy.run_module("src.scripts.run_feature_engineering", run_name="__main__")
    finally:
        sys.argv = orig_argv

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='lineup_trends'"
        )
        assert cur.fetchone() is not None
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pitcher_workload_features'"
        )
        assert cur.fetchone() is not None
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='model_features'"
        )
        assert cur.fetchone() is not None
        lineup_cols = [
            row[1] for row in conn.execute("PRAGMA table_info(lineup_trends)")
        ]
        assert "lineup_avg_ops_mean_3" in lineup_cols
        assert "projected_lineup_k_pct" in lineup_cols
        model_cols = [
            row[1] for row in conn.execute("PRAGMA table_info(model_features)")
        ]
        assert "lineup_avg_ops_mean_3" in model_cols
        assert "catcher_called_strike_rate_mean_3" in model_cols
        assert "projected_lineup_k_pct" in model_cols


def test_extra_cat_cols_excluded(tmp_path: Path) -> None:
    """Ensure problematic categorical columns are not mean-encoded."""
    db_path = setup_test_db(tmp_path)

    engineer_pitcher_features(db_path=db_path)
    engineer_opponent_features(db_path=db_path)
    engineer_contextual_features(db_path=db_path)

    # Inject columns that should not be encoded
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM contextual_features", conn)
        df["away_pitcher_ids"] = ["[1]"] * len(df)
        df["home_pitcher_ids"] = ["[2]"] * len(df)
        df["scraped_timestamp"] = "2024-04-01"
        df.to_sql("contextual_features", conn, if_exists="replace", index=False)

    engineer_lineup_trends(db_path=db_path)
    engineer_batter_pitcher_history(db_path=db_path)
    build_model_features(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM model_features", conn)
        assert "away_pitcher_ids_enc" not in df.columns
        assert "home_pitcher_ids_enc" not in df.columns
        assert "scraped_timestamp_enc" not in df.columns
        assert "away_pitcher_ids" not in df.columns
        assert "home_pitcher_ids" not in df.columns
        assert "scraped_timestamp" not in df.columns
