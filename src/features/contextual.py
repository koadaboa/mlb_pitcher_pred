from __future__ import annotations

import os
import re
from typing import List, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.utils import DBConnection, setup_logger
from src.config import DBConfig, StrikeoutModelConfig, LogConfig
from .engineer_features import _trend

logger = setup_logger(
    "contextual_features",
    LogConfig.LOG_DIR / "contextual_features.log",
)


TEAM_TO_BALLPARK = {
    "ARI": "Chase Field",
    "ATL": "Truist Park",
    "BAL": "Oriole Park at Camden Yards",
    "BOS": "Fenway Park",
    "CHC": "Wrigley Field",
    "CWS": "Guaranteed Rate Field",
    "CIN": "Great American Ball Park",
    "CLE": "Progressive Field",
    "COL": "Coors Field",
    "DET": "Comerica Park",
    "HOU": "Minute Maid Park",
    "KC": "Kauffman Stadium",
    "LAA": "Angel Stadium",
    "LAD": "Dodger Stadium",
    "MIA": "loanDepot Park",
    "MIL": "American Family Field",
    "MIN": "Target Field",
    "NYM": "Citi Field",
    "NYY": "Yankee Stadium",
    "OAK": "Oakland Coliseum",
    "PHI": "Citizens Bank Park",
    "PIT": "PNC Park",
    "SD": "Petco Park",
    "SF": "Oracle Park",
    "SEA": "T-Mobile Park",
    "STL": "Busch Stadium",
    "TB": "Tropicana Field",
    "TEX": "Globe Life Field",
    "TOR": "Rogers Centre",
    "WSH": "Nationals Park",
}


def _parse_wind_speed(value: str | None) -> float:
    if not value or not isinstance(value, str):
        return np.nan
    m = re.search(r"(\d+)", value)
    return float(m.group(1)) if m else np.nan


def _add_group_rolling(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    date_col: str,
    prefix: str,
    windows: List[int] | None = None,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = StrikeoutModelConfig.WINDOW_SIZES

    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    df = df.sort_values(list(group_cols) + [date_col])
    exclude_cols = {"game_pk"}.union(set(group_cols))
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns if c not in exclude_cols
    ]

    def _calc_for_col(col: str, local_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling stats for a single column using a dataframe slice."""
        grouped = local_df.groupby(list(group_cols))[col]
        shifted = grouped.shift(1)
        parts = []
        for window in windows:
            roll = shifted.rolling(window, min_periods=1)
            mean = roll.mean()
            stats = pd.DataFrame(
                {
                    f"{prefix}{col}_mean_{window}": mean,
                    f"{prefix}{col}_std_{window}": roll.std(),
                    f"{prefix}{col}_min_{window}": roll.min(),
                    f"{prefix}{col}_max_{window}": roll.max(),
                    f"{prefix}{col}_trend_{window}": roll.apply(_trend, raw=True),
                }
            )
            stats[f"{prefix}{col}_momentum_{window}"] = local_df[col] - mean
            parts.append(stats)
        return pd.concat(parts, axis=1)

    frames = [df]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_calc_for_col)(c, df[[c, *group_cols, date_col]]) for c in numeric_cols
    )
    frames.extend(results)

    df = pd.concat(frames, axis=1)
    return df


def engineer_opponent_features(
    db_path: str | None = None,
    source_table: str = "game_level_matchup_details",
    target_table: str = "rolling_pitcher_vs_team",
    n_jobs: int | None = None,
) -> pd.DataFrame:
    db_path = db_path or DBConfig.PATH
    logger.info("Loading matchup data from %s", source_table)
    with DBConnection(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {source_table}", conn)

    if df.empty:
        logger.warning("No data found in %s", source_table)
        return df

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = _add_group_rolling(
        df,
        ["pitcher_id", "opponent_team"],
        "game_date",
        prefix="opp_",
        n_jobs=n_jobs,
    )

    with DBConnection(db_path) as conn:
        df.to_sql(target_table, conn, if_exists="replace", index=False)
    logger.info("Saved opponent features to %s", target_table)
    return df


def engineer_contextual_features(
    db_path: str | None = None,
    source_table: str = "game_level_matchup_details",
    target_table: str = "contextual_features",
    n_jobs: int | None = None,
) -> pd.DataFrame:
    db_path = db_path or DBConfig.PATH
    logger.info("Loading matchup data from %s", source_table)
    with DBConnection(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {source_table}", conn)

    if df.empty:
        logger.warning("No data found in %s", source_table)
        return df

    df["game_date"] = pd.to_datetime(df["game_date"])
    if "temp" in df.columns:
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    if "wind" in df.columns:
        df["wind_speed"] = df["wind"].apply(_parse_wind_speed)
    if "elevation" in df.columns:
        df["elevation"] = pd.to_numeric(df["elevation"], errors="coerce")

    df = _add_group_rolling(
        df, ["hp_umpire"], "game_date", prefix="ump_", n_jobs=n_jobs
    )
    if "weather" in df.columns:
        df = _add_group_rolling(
            df, ["weather"], "game_date", prefix="wx_", n_jobs=n_jobs
        )
    df = _add_group_rolling(
        df, ["home_team"], "game_date", prefix="venue_", n_jobs=n_jobs
    )

    df["stadium"] = df["home_team"].map(TEAM_TO_BALLPARK)

    with DBConnection(db_path) as conn:
        df.to_sql(target_table, conn, if_exists="replace", index=False)
    logger.info("Saved contextual features to %s", target_table)
    return df
