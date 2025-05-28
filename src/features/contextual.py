from __future__ import annotations

import os
import re
from typing import List, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src.utils import (
    DBConnection,
    setup_logger,
    table_exists,
    get_latest_date,
)
from src.config import DBConfig, StrikeoutModelConfig, LogConfig

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
    numeric_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute rolling stats for specified groups.

    Parameters
    ----------
    df : DataFrame
        Input data containing all columns.
    group_cols : Sequence[str]
        Columns used to group consecutive games.
    date_col : str
        Column containing the chronological order of games.
    prefix : str
        Prefix for the generated feature names.
    windows : list[int], optional
        Rolling window sizes. Defaults to ``StrikeoutModelConfig.WINDOW_SIZES``.
    numeric_cols : Sequence[str], optional
        Restrict calculations to these numeric columns. If ``None`` (default),
        all numeric columns except identifiers are used.
    """
    if windows is None:
        windows = StrikeoutModelConfig.WINDOW_SIZES

    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    df = df.sort_values(list(group_cols) + [date_col])
    exclude_cols = {"game_pk"}.union(set(group_cols))
    if numeric_cols is None:
        numeric_cols = [
            c
            for c in df.select_dtypes(include=np.number).columns
            if c not in exclude_cols
        ]
    else:
        numeric_cols = [
            c for c in numeric_cols if c in df.columns and c not in exclude_cols
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
                }
            )
            stats[f"{prefix}{col}_momentum_{window}"] = shifted - mean
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
    year: int | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Compute rolling opponent statistics for each pitcher/team matchup.

    Parameters
    ----------
    rebuild : bool, default False
        When ``True`` the ``target_table`` is dropped before new rows are
        inserted so only features using the current window sizes remain.
    """

    db_path = db_path or DBConfig.PATH
    logger.info("Loading matchup data from %s", source_table)
    max_window = max(StrikeoutModelConfig.WINDOW_SIZES)
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

        if df.empty:
            logger.warning("No data found in %s", source_table)
            return df

        df["game_date"] = pd.to_datetime(df["game_date"])
        if latest is not None:
            df = df[df["game_date"] > latest]
        if df.empty:
            logger.info("No new rows to process for %s", target_table)
            return df
        df = _add_group_rolling(
            df,
            ["pitcher_id", "opponent_team"],
            "game_date",
            prefix="opp_",
            n_jobs=n_jobs,
            numeric_cols=StrikeoutModelConfig.PITCHER_ROLLING_COLS,
        )
        if rebuild or not table_exists(conn, target_table):
            df.to_sql(target_table, conn, if_exists="replace", index=False)
        else:
            df.to_sql(target_table, conn, if_exists="append", index=False)
        logger.info("Saved opponent features to %s", target_table)
        return df


def engineer_contextual_features(
    db_path: str | None = None,
    source_table: str = "game_level_matchup_details",
    target_table: str = "contextual_features",
    n_jobs: int | None = None,
    year: int | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Aggregate contextual factors and compute rolling statistics.

    Parameters
    ----------
    rebuild : bool, default False
        Drop and recreate ``target_table`` so outdated window sizes are removed.
    """

    db_path = db_path or DBConfig.PATH
    logger.info("Loading matchup data from %s", source_table)
    max_window = max(StrikeoutModelConfig.WINDOW_SIZES)
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

        if df.empty:
            logger.warning("No data found in %s", source_table)
            return df

        df["game_date"] = pd.to_datetime(df["game_date"])
        if latest is not None:
            df = df[df["game_date"] > latest]
        if df.empty:
            logger.info("No new rows to process for %s", target_table)
            return df

        if "temp" in df.columns:
            df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        if "wind" in df.columns:
            df["wind_speed"] = df["wind"].apply(_parse_wind_speed)
        if "elevation" in df.columns:
            df["elevation"] = pd.to_numeric(df["elevation"], errors="coerce")

        df = _add_group_rolling(
            df,
            ["hp_umpire"],
            "game_date",
            prefix="ump_",
            n_jobs=n_jobs,
            numeric_cols=StrikeoutModelConfig.CONTEXT_ROLLING_COLS,
        )
        if "weather" in df.columns:
            df = _add_group_rolling(
                df,
                ["weather"],
                "game_date",
                prefix="wx_",
                n_jobs=n_jobs,
                numeric_cols=StrikeoutModelConfig.CONTEXT_ROLLING_COLS,
            )
        df = _add_group_rolling(
            df,
            ["home_team"],
            "game_date",
            prefix="venue_",
            n_jobs=n_jobs,
            numeric_cols=StrikeoutModelConfig.CONTEXT_ROLLING_COLS,
        )

        df["stadium"] = df["home_team"].map(TEAM_TO_BALLPARK)

        if rebuild or not table_exists(conn, target_table):
            df.to_sql(target_table, conn, if_exists="replace", index=False)
        else:
            df.to_sql(target_table, conn, if_exists="append", index=False)
        logger.info("Saved contextual features to %s", target_table)
        return df
