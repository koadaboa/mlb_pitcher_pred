from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import logging

from typing import Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from src.utils import DBConnection, setup_logger
from src.config import DBConfig, LogConfig

# --- Pitch Type Groups ---
FASTBALL_TYPES = {"FF", "FA", "FT", "SI", "F4", "F2", "FC", "FS", "SF", "FO"}
BREAKING_TYPES = {"SL", "CU", "KC", "SV", "SC"}
OFFSPEED_TYPES = {"CH", "FO", "KN", "EP"}

logger = setup_logger(
    "create_starting_pitcher_table",
    LogConfig.LOG_DIR / "create_starting_pitcher_table.log",
)

# Log progress after this many games have been processed
LOG_EVERY_N = 100

# Number of worker processes for parallel aggregation
MAX_WORKERS = 9
# Submit this many tasks to the pool at a time
BATCH_SIZE = 5000

def filter_starting_pitchers(conn) -> pd.DataFrame:
    """Return game_pk/pitcher combos likely representing true starters."""
    query = """
        SELECT game_pk, pitcher
        FROM statcast_pitchers
        GROUP BY game_pk, pitcher
        HAVING MIN(inning) = 1
           AND COUNT(*) > 30
           AND MAX(inning) > 3
    """
    df = pd.read_sql_query(query, conn)
    logger.info("Found %d potential starting pitcher rows", len(df))
    return df


def load_pitcher_game(conn, game_pk: int, pitcher: int) -> pd.DataFrame:
    """Load all pitch-level rows for a pitcher in one game."""
    q = "SELECT * FROM statcast_pitchers WHERE game_pk = ? AND pitcher = ?"
    return pd.read_sql_query(q, conn, params=(game_pk, pitcher))

def compute_game_features(game_pk: int, pitcher: int, db_path: Path) -> Optional[Dict]:
    """Load one pitcher/game from SQLite and compute features."""
    with DBConnection(db_path) as conn:
        df = load_pitcher_game(conn, game_pk, pitcher)
    if df.empty:
        return None
    return compute_features(df)

def compute_features(df: pd.DataFrame) -> Dict:
    df = df.sort_values(["at_bat_number", "pitch_number"])
    first_row = df.iloc[0]
    if first_row["inning_topbot"] == "Top":
        team = first_row["home_team"]
        opp = first_row["away_team"]
    else:
        team = first_row["away_team"]
        opp = first_row["home_team"]

    pitches = len(df)
    strike_events = df["events"].isin(["strikeout", "strikeout_double_play"])
    swinging = df["description"].str.contains("swinging_strike", na=False)
    called = df["description"].eq("called_strike")
    foul_tip = df["description"].eq("foul_tip")

    first_pitch = df[df["pitch_number"] == 1]
    first_pitch_strikes = first_pitch["type"].eq("S")

    fastball_mask = df["pitch_type"].isin(FASTBALL_TYPES)
    breaking_mask = df["pitch_type"].isin(BREAKING_TYPES)
    offspeed_mask = df["pitch_type"].isin(OFFSPEED_TYPES)

    # Use mutually exclusive pitch type sets for ratio calculations
    fastball_only = df["pitch_type"].isin(FASTBALL_TYPES - OFFSPEED_TYPES)
    offspeed_only = df["pitch_type"].isin(OFFSPEED_TYPES - FASTBALL_TYPES)

    types = df["pitch_type"].values
    next_types = np.roll(types, -1)
    fastball_then_break = fastball_mask & np.isin(next_types, list(BREAKING_TYPES))

    features = {
        "game_pk": first_row["game_pk"],
        "game_date": first_row["game_date"],
        "pitcher_id": first_row["pitcher"],
        "pitcher_hand": first_row["p_throws"],
        "pitching_team": team,
        "opponent_team": opp,
        "pitches": pitches,
        "innings_pitched": df["inning"].nunique(),
        "batters_faced": df["batter"].nunique(),
        "strikeouts": strike_events.sum(),
        "swinging_strike_rate": swinging.mean(),
        "first_pitch_strike_rate": (
            first_pitch_strikes.mean() if len(first_pitch) else np.nan
        ),
        "csw_pct": ((called | swinging | foul_tip).mean()),
        "fastball_pct": fastball_only.mean(),
        "offspeed_to_fastball_ratio": (
            offspeed_only.sum() / fastball_only.sum() if fastball_only.sum() else np.nan
        ),
        "fastball_then_breaking_rate": (
            fastball_then_break[:-1].mean() if len(df) > 1 else np.nan
        ),
        "avg_release_speed": df["release_speed"].mean(),
        "max_release_speed": df["release_speed"].max(),
        "avg_spin_rate": df["release_spin_rate"].mean(),
        "unique_pitch_types": df["pitch_type"].nunique(),
    }
    return features


def aggregate_to_game_level(db_path: Path = DBConfig.PATH) -> pd.DataFrame:
    with DBConnection(db_path) as conn:
        starters = filter_starting_pitchers(conn)
        total_games = len(starters)

    result_rows: list[Dict] = []
    processed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exc:
        pending = []
        for game_pk, pitcher in starters.itertuples(index=False):
            pending.append(exc.submit(compute_game_features, game_pk, pitcher, db_path))
            if len(pending) >= BATCH_SIZE:
                for fut in as_completed(pending):
                    res = fut.result()
                    processed += 1
                    if res:
                        result_rows.append(res)
                    if processed % LOG_EVERY_N == 0:
                        logger.info("Processed %d/%d games", processed, total_games)
                pending.clear()

        for fut in as_completed(pending):
            res = fut.result()
            processed += 1
            if res:
                result_rows.append(res)
            if processed % LOG_EVERY_N == 0:
                logger.info("Processed %d/%d games", processed, total_games)

    game_df = pd.DataFrame(result_rows)
    with DBConnection(db_path) as conn:
        game_df.to_sql(
            "game_level_starting_pitchers",
            conn,
            index=False,
            if_exists="replace",
        )
    return game_df


def main() -> None:
    try:
        df = aggregate_to_game_level()
        logger.info(f"Aggregated {len(df)} starting pitcher games")
    except Exception as exc:
        logger.exception("Failed to create starting pitcher table: %s", exc)


if __name__ == "__main__":
    main()
