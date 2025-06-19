from __future__ import annotations

import pandas as pd
from pathlib import Path
import logging

from src.utils import (
    DBConnection,
    setup_logger,
    safe_merge,
    parse_starting_pitcher_id,
)
from src.config import DBConfig, LogConfig

STARTERS_TABLE = "game_level_starting_pitchers"
TEAM_BATTING_TABLE = "game_level_team_batting"
BOXSCORES_TABLE = "mlb_boxscores"
OUTPUT_TABLE = "game_level_matchup_details"

logger = setup_logger(
    "create_matchup_details",
    LogConfig.LOG_DIR / "create_matchup_details.log",
)


def build_matchup_table(db_path: Path = DBConfig.PATH) -> pd.DataFrame:
    """Join starting pitcher, team batting, and boxscore tables."""
    with DBConnection(db_path) as conn:
        starters = pd.read_sql_query(f"SELECT * FROM {STARTERS_TABLE}", conn)
        if starters.empty:
            logger.warning("No rows found in %s", STARTERS_TABLE)
            return pd.DataFrame()

        team_bat = pd.read_sql_query(f"SELECT * FROM {TEAM_BATTING_TABLE}", conn)
        if team_bat.empty:
            logger.warning("No rows found in %s", TEAM_BATTING_TABLE)
        boxscores = pd.read_sql_query(f"SELECT * FROM {BOXSCORES_TABLE}", conn)
        if boxscores.empty:
            logger.warning("No rows found in %s", BOXSCORES_TABLE)
        else:
            if "away_pitcher_ids" in boxscores.columns:
                boxscores["away_starting_pitcher_id"] = boxscores[
                    "away_pitcher_ids"
                ].apply(parse_starting_pitcher_id)
            if "home_pitcher_ids" in boxscores.columns:
                boxscores["home_starting_pitcher_id"] = boxscores[
                    "home_pitcher_ids"
                ].apply(parse_starting_pitcher_id)

        # Merge starter metrics with opponent batting
        merge_cols = ["game_pk", "pitcher_id", "opponent_team"]
        cols_in_team = set(team_bat.columns)
        if not set(merge_cols).issubset(cols_in_team):
            merge_cols = ["game_pk", "pitching_team", "opponent_team"]
        merged = safe_merge(starters, team_bat, on=merge_cols, how="left")

        # Add boxscore information (joined on game_pk) and preserve existing
        # columns (including game_date) without automatic suffixing.
        merged = safe_merge(
            merged,
            boxscores,
            on="game_pk",
            how="left",
            suffixes=("", "_bx"),
        )

        # If the boxscores table also contains a ``game_date`` column, fill any
        # missing values in the starter data and drop the extra column so the
        # final table has a single ``game_date`` field.
        if "game_date_bx" in merged.columns:
            merged["game_date"] = merged["game_date"].fillna(merged["game_date_bx"])
            merged = merged.drop(columns=["game_date_bx"])
        merged.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)
        logger.info("Wrote %d rows to %s", len(merged), OUTPUT_TABLE)
        return merged


def main() -> None:
    try:
        df = build_matchup_table()
        logger.info("Created matchup table with %d rows", len(df))
    except Exception as exc:
        logger.exception("Failed to create matchup table: %s", exc)


if __name__ == "__main__":
    main()
