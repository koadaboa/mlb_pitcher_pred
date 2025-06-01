from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import DBConnection, setup_logger, table_exists, get_latest_date
from src.config import DBConfig, LogConfig

logger = setup_logger("workload_features", LogConfig.LOG_DIR / "workload_features.log")


def add_recent_pitch_counts(df: pd.DataFrame, window_days: int = 7) -> pd.Series:
    """Return rolling sum of pitches thrown in the previous ``window_days``."""
    df = df.sort_values(["pitcher_id", "game_date"])
    result_parts = []
    for pid, g in df.groupby("pitcher_id"):
        shifted = g.set_index("game_date")["pitches"].shift(1)
        rolled = shifted.rolling(f"{window_days}D").sum()
        # Maintain alignment with the original group by restoring its index
        rolled.index = g.index
        result_parts.append(rolled)
    out = pd.concat(result_parts).sort_index()
    out.name = f"pitches_last_{window_days}d"
    return out


def add_injury_indicators(df: pd.DataFrame, injury_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``on_il`` and ``days_since_il`` columns using ``player_injury_log``."""
    if injury_df.empty:
        df["on_il"] = 0
        df["days_since_il"] = np.nan
        return df

    injury_df = injury_df.copy()
    injury_df["start_date"] = pd.to_datetime(injury_df["start_date"])
    injury_df["end_date"] = pd.to_datetime(injury_df["end_date"])
    df = df.sort_values(["pitcher_id", "game_date"])

    frames = []
    for pid, g in df.groupby("pitcher_id"):
        stints = injury_df[injury_df["player_id"] == pid].sort_values("start_date")
        if stints.empty:
            g["on_il"] = 0
            g["days_since_il"] = np.nan
            frames.append(g)
            continue

        last_end = pd.NaT
        on_il = []
        days_since = []
        for date in g["game_date"]:
            current = stints[(stints["start_date"] <= date) & ((stints["end_date"].isna()) | (date < stints["end_date"]))]
            on_il.append(1 if not current.empty else 0)
            past = stints[stints["end_date"] <= date]
            if not past.empty:
                last_end = past["end_date"].max()
            days_since.append((date - last_end).days if pd.notna(last_end) else np.nan)
        g["on_il"] = on_il
        g["days_since_il"] = days_since
        frames.append(g)

    return pd.concat(frames).sort_index()


def engineer_workload_features(
    db_path: Path = DBConfig.PATH,
    source_table: str = "game_level_starting_pitchers",
    injury_table: str = "player_injury_log",
    target_table: str = "pitcher_workload_features",
    year: int | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Compute workload and injury features for starting pitchers."""
    with DBConnection(db_path) as conn:
        if rebuild and table_exists(conn, target_table):
            conn.execute(f"DROP TABLE IF EXISTS {target_table}")
            latest = None
        else:
            latest = get_latest_date(conn, target_table, "game_date")

        query = f"SELECT * FROM {source_table}"
        if year:
            query += f" WHERE strftime('%Y', game_date) = '{year}'"
        df = pd.read_sql_query(query, conn)

        if table_exists(conn, injury_table):
            injury_df = pd.read_sql_query(f"SELECT * FROM {injury_table}", conn)
        else:
            injury_df = pd.DataFrame(columns=["player_id", "start_date", "end_date"])

    if df.empty:
        logger.warning("No data found in %s", source_table)
        return df

    df["game_date"] = pd.to_datetime(df["game_date"])
    if latest is not None:
        df = df[df["game_date"] > latest]
    if df.empty:
        logger.info("No new rows to process for %s", target_table)
        return df

    df[f"pitches_last_7d"] = add_recent_pitch_counts(df, 7)
    df = add_injury_indicators(df, injury_df)

    with DBConnection(db_path) as conn:
        if rebuild or not table_exists(conn, target_table):
            df.to_sql(target_table, conn, index=False, if_exists="replace")
        else:
            df.to_sql(target_table, conn, index=False, if_exists="append")
    logger.info("Saved workload features to %s", target_table)
    return df
