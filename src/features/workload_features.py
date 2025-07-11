from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3

from src.utils import (
    DBConnection,
    setup_logger,
    table_exists,
    get_latest_date,
    safe_merge,
    load_table_cached,
)
from src.config import DBConfig, LogConfig

logger = setup_logger("workload_features", LogConfig.LOG_DIR / "workload_features.log")


def add_recent_pitch_counts(df: pd.DataFrame, window_days: int = 7) -> pd.Series:
    """Return rolling sum of pitches thrown in the previous ``window_days``."""

    # Compute rolling sums grouped by pitcher without explicit Python loops
    df_sorted = df.sort_values(["pitcher_id", "game_date"])
    rolled = (
        df_sorted.set_index(["pitcher_id", "game_date"])
        .groupby(level=0)["pitches"]
        .shift(1)
        .rolling(f"{window_days}D")
        .sum()
    )

    # Align results with the sorted dataframe and restore original row order
    rolled.index = df_sorted.index
    rolled = rolled.reindex(df.index)
    rolled.name = f"pitches_last_{window_days}d"
    return rolled


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
            current = stints[
                (stints["start_date"] <= date)
                & ((stints["end_date"].isna()) | (date < stints["end_date"]))
            ]
            on_il.append(1 if not current.empty else 0)
            past = stints[stints["end_date"] <= date]
            if not past.empty:
                last_end = past["end_date"].max()
            days_since.append((date - last_end).days if pd.notna(last_end) else np.nan)
        g["on_il"] = on_il
        g["days_since_il"] = days_since
        frames.append(g)

    return pd.concat(frames).sort_index()


def add_pitcher_age(df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    """Merge birth dates and calculate pitcher age on ``game_date``."""
    if players_df.empty or "birth_date" not in players_df.columns:
        df["pitcher_age"] = np.nan
        return df

    players_df = players_df.rename(columns={"player_id": "pitcher_id"})
    players_df["birth_date"] = pd.to_datetime(players_df["birth_date"])
    df = safe_merge(
        df, players_df[["pitcher_id", "birth_date"]], on="pitcher_id", how="left"
    )
    df["pitcher_age"] = (df["game_date"] - df["birth_date"]).dt.days / 365.25
    df = df.drop(columns=["birth_date"], errors="ignore")
    return df


def add_recent_innings(df: pd.DataFrame, window_days: int = 30) -> pd.Series:
    """Return rolling sum of innings pitched in the previous ``window_days``."""

    df_sorted = df.sort_values(["pitcher_id", "game_date"])
    rolled = (
        df_sorted.set_index(["pitcher_id", "game_date"])
        .groupby(level=0)["innings_pitched"]
        .shift(1)
        .rolling(f"{window_days}D")
        .sum()
    )

    rolled.index = df_sorted.index
    rolled = rolled.reindex(df.index)
    rolled.name = f"season_ip_last_{window_days}d"
    return rolled


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

        supports_window = sqlite3.sqlite_version_info >= (3, 25, 0)
        if supports_window:
            pitch_expr = (
                "SUM(pitches) OVER (PARTITION BY pitcher_id ORDER BY game_date "
                "ROWS BETWEEN ? PRECEDING AND 1 PRECEDING) AS pitches_last_7d"
            )
            ip_expr = (
                "SUM(innings_pitched) OVER (PARTITION BY pitcher_id ORDER BY game_date "
                "ROWS BETWEEN ? PRECEDING AND 1 PRECEDING) AS season_ip_last_30d"
            )
            query = f"SELECT *, {pitch_expr}, {ip_expr} FROM {source_table}"
            params = (7, 30)
        else:
            query = f"SELECT * FROM {source_table}"
            params = ()

        if year:
            query += f" WHERE strftime('%Y', game_date) = '{year}'"

        df = pd.read_sql_query(query, conn, params=params)

        if table_exists(conn, injury_table):
            injury_df = load_table_cached(db_path, injury_table, rebuild=rebuild)
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

    if supports_window:
        # Columns were calculated in SQL query
        pass
    else:
        df["pitches_last_7d"] = add_recent_pitch_counts(df, 7)
        df["season_ip_last_30d"] = add_recent_innings(df, 30)
    df = add_injury_indicators(df, injury_df)

    with DBConnection(db_path) as conn:
        if rebuild or not table_exists(conn, target_table):
            df.to_sql(target_table, conn, index=False, if_exists="replace")
        else:
            df.to_sql(target_table, conn, index=False, if_exists="append")
    logger.info("Saved workload features to %s", target_table)
    return df
