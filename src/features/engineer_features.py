from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import logging

from typing import List

from src.utils import DBConnection, setup_logger
from src.config import DBConfig, StrikeoutModelConfig, LogConfig

logger = setup_logger(
    "engineer_features",
    LogConfig.LOG_DIR / "engineer_features.log",
)


def _trend(values: np.ndarray) -> float:
    """Return the slope of a simple linear regression for the input values."""
    if len(values) < 2:
        return np.nan
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """Add rolling stats, momentum, and trend features to ``df``.

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
    """
    if windows is None:
        windows = StrikeoutModelConfig.WINDOW_SIZES

    df = df.sort_values([group_col, date_col])
    numeric_cols = [
        c
        for c in df.select_dtypes(include=np.number).columns
        if c not in {"game_pk", group_col}
    ]

    for col in numeric_cols:
        grouped = df.groupby(group_col)[col]
        shifted = grouped.shift(1)
        for window in windows:
            roll = shifted.rolling(window, min_periods=1)
            mean_col = f"{col}_mean_{window}"
            df[mean_col] = roll.mean()
            df[f"{col}_std_{window}"] = roll.std()
            df[f"{col}_min_{window}"] = roll.min()
            df[f"{col}_max_{window}"] = roll.max()
            df[f"{col}_trend_{window}"] = roll.apply(_trend, raw=True)
            df[f"{col}_momentum_{window}"] = df[col] - df[mean_col]
    return df


def engineer_pitcher_features(
    db_path: Path = DBConfig.PATH,
    source_table: str = "game_level_starting_pitchers",
    target_table: str = "rolling_pitcher_features",
) -> pd.DataFrame:
    """Load pitcher game stats, compute rolling features, and store the result."""
    logger.info("Loading data from %s", source_table)
    with DBConnection(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {source_table}", conn)

    if "game_date" not in df.columns:
        logger.error("Required column 'game_date' not found in %s", source_table)
        raise KeyError("game_date not found")

    df["game_date"] = pd.to_datetime(df["game_date"])
    logger.info("Computing rolling features for %d rows", len(df))
    df = add_rolling_features(df, group_col="pitcher_id", date_col="game_date")

    with DBConnection(db_path) as conn:
        df.to_sql(target_table, conn, if_exists="replace", index=False)
    logger.info("Saved features to table '%s'", target_table)
    return df


def main() -> None:
    try:
        engineer_pitcher_features()
    except Exception as exc:
        logger.exception("Failed to engineer features: %s", exc)


if __name__ == "__main__":
    main()