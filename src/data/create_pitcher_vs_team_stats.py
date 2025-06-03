from __future__ import annotations

import pandas as pd
from pathlib import Path
import logging

from src.utils import DBConnection, setup_logger, safe_merge
from src.config import DBConfig, LogConfig

STARTERS_TABLE = "game_level_starting_pitchers"
TEAM_BATTING_TABLE = "game_level_team_batting"
OUTPUT_TABLE = "game_level_matchup_stats"

logger = setup_logger(
    "create_pitcher_vs_team_stats",
    LogConfig.LOG_DIR / "create_pitcher_vs_team_stats.log",
)


def build_matchup_stats(db_path: Path = DBConfig.PATH) -> pd.DataFrame:
    """Join starting pitcher stats with opponent team metrics."""
    with DBConnection(db_path) as conn:
        starters = pd.read_sql_query(f"SELECT * FROM {STARTERS_TABLE}", conn)
        if starters.empty:
            logger.warning("No rows found in %s", STARTERS_TABLE)
            return pd.DataFrame()

        team_bat = pd.read_sql_query(f"SELECT * FROM {TEAM_BATTING_TABLE}", conn)
        if team_bat.empty:
            logger.warning("No rows found in %s", TEAM_BATTING_TABLE)

        merge_cols = ["game_pk", "pitcher_id", "opponent_team"]
        if not set(merge_cols).issubset(team_bat.columns):
            merge_cols = ["game_pk", "pitching_team", "opponent_team"]

        merged = safe_merge(starters, team_bat, on=merge_cols, how="left")
        merged.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)
        logger.info("Wrote %d rows to %s", len(merged), OUTPUT_TABLE)
        return merged


def main() -> None:
    try:
        df = build_matchup_stats()
        logger.info("Created matchup stats with %d rows", len(df))
    except Exception as exc:
        logger.exception("Failed to create matchup stats: %s", exc)


if __name__ == "__main__":
    main()
