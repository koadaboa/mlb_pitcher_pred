from __future__ import annotations

import os
import re
from typing import List, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.utils import DBConnection, setup_logger, get_latest_date
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
    max_window = max(StrikeoutModelConfig.WINDOW_SIZES)
    with DBConnection(db_path) as conn:
        latest = get_latest_date(conn, target_table)
        if latest is not None:
            logger.info("Existing opponent features up to %s", latest.date())
            prev = pd.read_sql_query(
                f"SELECT * FROM {source_table} WHERE game_date <= ? ORDER BY pitcher_id, opponent_team, game_date DESC",
                conn,
                params=(latest,),
            )
            prev = prev.groupby(["pitcher_id", "opponent_team"]).head(max_window)
            new_rows = pd.read_sql_query(
                f"SELECT * FROM {source_table} WHERE game_date > ?",
                conn,
                params=(latest,),
            )
            if new_rows.empty:
                logger.info("No new games to process")
                return pd.DataFrame()
            df = pd.concat([prev, new_rows], ignore_index=True)
        else:
            df = pd.read_sql_query(f"SELECT * FROM {source_table}", conn)

    if df.empty:
        logger.warning("No data found in %s", source_table)
        return df

    df["game_date"] = pd.to_datetime(df["game_date"])
    features = _add_group_rolling(
        df,
        ["pitcher_id", "opponent_team"],
        "game_date",
        prefix="opp_",
        n_jobs=n_jobs,
    )

    if latest is not None:
        new_features = features[features["game_date"] > latest]
        if new_features.empty:
            logger.info("No new opponent features generated")
            return pd.DataFrame()
        with DBConnection(db_path) as conn:
            new_features.to_sql(target_table, conn, if_exists="append", index=False)
        logger.info("Appended %d new rows to %s", len(new_features), target_table)
        return new_features
    else:
        with DBConnection(db_path) as conn:
            features.to_sql(target_table, conn, if_exists="replace", index=False)
        logger.info("Saved opponent features to %s", target_table)
        return features


def engineer_contextual_features(
    db_path: str | None = None,
    source_table: str = "game_level_matchup_details",
    target_table: str = "contextual_features",
    n_jobs: int | None = None,
) -> pd.DataFrame:
    db_path = db_path or DBConfig.PATH
    logger.info("Loading matchup data from %s", source_table)
    max_window = max(StrikeoutModelConfig.WINDOW_SIZES)
    with DBConnection(db_path) as conn:
        latest = get_latest_date(conn, target_table)
        if latest is not None:
            logger.info("Existing contextual features up to %s", latest.date())
            prev = pd.read_sql_query(
                f"SELECT * FROM {source_table} WHERE game_date <= ? ORDER BY pitcher_id, game_date DESC",
                conn,
                params=(latest,),
            )
            prev = prev.groupby("pitcher_id").head(max_window)
            new_rows = pd.read_sql_query(
                f"SELECT * FROM {source_table} WHERE game_date > ?",
                conn,
                params=(latest,),
            )
            if new_rows.empty:
                logger.info("No new games to process")
                return pd.DataFrame()
            df = pd.concat([prev, new_rows], ignore_index=True)
        else:
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

    features = _add_group_rolling(
        df, ["hp_umpire"], "game_date", prefix="ump_", n_jobs=n_jobs
    )
    if "weather" in df.columns:
        features = _add_group_rolling(
            features, ["weather"], "game_date", prefix="wx_", n_jobs=n_jobs
        )
    features = _add_group_rolling(
        features, ["home_team"], "game_date", prefix="venue_", n_jobs=n_jobs
    )

    features["stadium"] = features["home_team"].map(TEAM_TO_BALLPARK)

    if latest is not None:
        new_features = features[features["game_date"] > latest]
        if new_features.empty:
            logger.info("No new contextual features generated")
            return pd.DataFrame()
        with DBConnection(db_path) as conn:
            new_features.to_sql(target_table, conn, if_exists="append", index=False)
        logger.info("Appended %d new rows to %s", len(new_features), target_table)
        return new_features
    else:
        with DBConnection(db_path) as conn:
            features.to_sql(target_table, conn, if_exists="replace", index=False)
        logger.info("Saved contextual features to %s", target_table)
        return features
