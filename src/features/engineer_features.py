from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import logging

from typing import List, Sequence

from src.utils import (
    DBConnection,
    setup_logger,
    table_exists,
    get_latest_date,
    load_table_cached,
)
from src.config import DBConfig, StrikeoutModelConfig, LogConfig
from .workload_features import (
    add_recent_pitch_counts,
    add_injury_indicators,
    add_pitcher_age,
    add_recent_innings,
)

logger = setup_logger(
    "engineer_features",
    LogConfig.LOG_DIR / "engineer_features.log",
)


def calculate_rest_days(
    df: pd.DataFrame,
    group_col: str = "pitcher_id",
    date_col: str = "game_date",
) -> pd.Series:
    """Return days between consecutive appearances for each ``group_col``.

    Parameters
    ----------
    df : DataFrame
        Input dataframe containing ``group_col`` and ``date_col``.
    group_col : str, default "pitcher_id"
        Column to group by when computing rest days.
    date_col : str, default "game_date"
        Column with chronological ordering of games.
    """

    ordered = df.sort_values([group_col, date_col])
    return ordered.groupby(group_col)[date_col].diff().dt.days


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    windows: List[int] | None = None,
    numeric_cols: Sequence[str] | None = None,
    ewm_halflife: float | None = None,
) -> pd.DataFrame:
    """Add rolling statistics and momentum features to ``df``.

    Parameters
    ----------
    df : DataFrame
        Input data sorted by ``group_col`` and ``date_col``.
    group_col : str
        Column used to group consecutive games (e.g., ``pitcher_id``).
    date_col : str
        Column containing the chronological order of games.
    windows : list[int], optional
        Rolling window sizes. Defaults to ``StrikeoutModelConfig.WINDOW_SIZES``.
    numeric_cols : Sequence[str], optional
        Limit calculations to these numeric columns. If ``None`` (default), use
        all numeric columns except identifiers.
    ewm_halflife : float, optional
        If provided, also compute exponentially weighted moving averages using
        ``halflife``. Columns ``<col>_ewm_<halflife>`` and
        ``<col>_momentum_ewm_<halflife>`` will be appended.
    """
    if windows is None:
        windows = StrikeoutModelConfig.WINDOW_SIZES

    df = df.sort_values([group_col, date_col])
    if numeric_cols is None:
        numeric_cols = [
            c
            for c in df.select_dtypes(include=np.number).columns
            if c not in {"game_pk", group_col}
        ]
    else:
        numeric_cols = [
            c
            for c in numeric_cols
            if c in df.columns and c not in {"game_pk", group_col}
        ]

    # remove any duplicate columns while preserving order
    seen = set()
    numeric_cols = [c for c in numeric_cols if not (c in seen or seen.add(c))]

    frames = [df]
    for col in numeric_cols:
        grouped = df.groupby(group_col)[col]
        shifted = grouped.shift(1)
        for window in windows:
            roll = shifted.groupby(df[group_col]).rolling(window, min_periods=1)
            mean = roll.mean().reset_index(level=0, drop=True)
            stats = pd.DataFrame(
                {
                    f"{col}_mean_{window}": mean,
                    f"{col}_std_{window}": roll.std().reset_index(level=0, drop=True),
                }
            )
            # Momentum compares last game's value to the previous average
            stats[f"{col}_momentum_{window}"] = shifted - mean
            frames.append(stats)
        if ewm_halflife is not None:
            ewm = grouped.apply(
                lambda x: x.shift(1).ewm(halflife=ewm_halflife, min_periods=1).mean()
            )
            ewm = ewm.reset_index(level=0, drop=True)
            ewm_stats = pd.DataFrame({f"{col}_ewm_{int(ewm_halflife)}": ewm})
            ewm_stats[f"{col}_momentum_ewm_{int(ewm_halflife)}"] = shifted - ewm
            frames.append(ewm_stats)

    df = pd.concat(frames, axis=1)
    return df


def engineer_pitcher_features(
    db_path: Path = DBConfig.PATH,
    source_table: str = "game_level_starting_pitchers",
    target_table: str = "rolling_pitcher_features",
    year: int | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Compute rolling pitcher features and append new rows to the database.

    Parameters
    ----------
    rebuild : bool, default False
        If ``True`` drop ``target_table`` before computing features so the table
        is recreated from scratch.
    """
    logger.info("Loading data from %s", source_table)
    with DBConnection(db_path) as conn:
        if rebuild and table_exists(conn, target_table):
            conn.execute(f"DROP TABLE IF EXISTS {target_table}")
            latest = None
        else:
            latest = get_latest_date(conn, target_table, "game_date")

        df = load_table_cached(db_path, source_table, year, rebuild=rebuild)

    if "game_date" not in df.columns:
        logger.error("Required column 'game_date' not found in %s", source_table)
        raise KeyError("game_date not found")

    df["game_date"] = pd.to_datetime(df["game_date"])
    if latest is not None:
        df = df[df["game_date"] > latest]

    if df.empty:
        logger.info("No new rows to process for %s", target_table)
        return df

    # Days into season relative to each year's first game
    df["season_year"] = df["game_date"].dt.year
    opening_day = df.groupby("season_year")["game_date"].transform("min")
    df["days_into_season"] = (df["game_date"] - opening_day).dt.days
    df.drop(columns=["season_year"], inplace=True)

    month = df["game_date"].dt.month
    df["month_bucket"] = pd.cut(
        month,
        bins=[0, 4, 8, 12],
        labels=["early", "mid", "late"],
        include_lowest=True,
    ).astype(str)


    df["rest_days"] = calculate_rest_days(df, "pitcher_id", "game_date")

    # Add workload features
    with DBConnection(db_path) as conn:
        if table_exists(conn, "player_injury_log"):
            injury_df = load_table_cached(db_path, "player_injury_log", rebuild=rebuild)
        else:
            injury_df = pd.DataFrame(columns=["player_id", "start_date", "end_date"])
        if table_exists(conn, "players"):
            player_df = load_table_cached(db_path, "players", rebuild=rebuild)
            if {
                "player_id",
                "birth_date",
            }.issubset(player_df.columns):
                player_df = player_df[["player_id", "birth_date"]]
        else:
            player_df = pd.DataFrame(columns=["player_id", "birth_date"])

    df["pitches_last_7d"] = add_recent_pitch_counts(df, 7)
    df["season_ip_last_30d"] = add_recent_innings(df, 30)
    df = add_injury_indicators(df, injury_df)
    df = add_pitcher_age(df, player_df)

    logger.info("Computing rolling features for %d rows", len(df))
    df = add_rolling_features(
        df,
        group_col="pitcher_id",
        date_col="game_date",
        windows=StrikeoutModelConfig.WINDOW_SIZES,
        numeric_cols=StrikeoutModelConfig.PITCHER_ROLLING_COLS,
        ewm_halflife=StrikeoutModelConfig.EWM_HALFLIFE,
    )

    with DBConnection(db_path) as conn:
        if rebuild or not table_exists(conn, target_table):
            df.to_sql(target_table, conn, if_exists="replace", index=False)
        else:
            df.to_sql(target_table, conn, if_exists="append", index=False)
    logger.info("Saved features to table '%s'", target_table)
    return df


def main() -> None:
    try:
        engineer_pitcher_features()
    except Exception as exc:
        logger.exception("Failed to engineer features: %s", exc)


if __name__ == "__main__":
    main()
