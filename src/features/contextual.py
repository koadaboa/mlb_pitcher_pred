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
    safe_merge,
    load_table_cached,
)
from src.config import (
    DBConfig,
    StrikeoutModelConfig,
    LogConfig,
    BALLPARK_FACTORS,
    BALLPARK_COORDS,
)

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


def _haversine_distance(
    coord1: tuple[float, float] | None, coord2: tuple[float, float] | None
) -> float:
    """Return great-circle distance in miles between two lat/lon coordinates."""
    if not coord1 or not coord2:
        return np.nan

    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_miles = 3958.8
    return float(earth_radius_miles * c)


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
    ewm_halflife: float | None = None,
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
    ewm_halflife : float, optional
        If provided, compute exponentially weighted moving averages using the
        specified ``halflife``. Columns are suffixed with ``ewm_<halflife>`` and
        ``momentum_ewm_<halflife>``.
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
            roll = shifted.groupby([local_df[c] for c in group_cols]).rolling(
                window, min_periods=1
            )
            mean = roll.mean().reset_index(
                level=list(range(len(group_cols))), drop=True
            )
            stats = pd.DataFrame(
                {
                    f"{prefix}{col}_mean_{window}": mean,
                    f"{prefix}{col}_std_{window}": roll.std().reset_index(
                        level=list(range(len(group_cols))), drop=True
                    ),
                }
            )
            stats[f"{prefix}{col}_momentum_{window}"] = shifted - mean
            parts.append(stats)
        if ewm_halflife is not None:
            ewm = grouped.apply(
                lambda x: x.shift(1).ewm(halflife=ewm_halflife, min_periods=1).mean()
            )
            ewm = ewm.reset_index(level=list(range(len(group_cols))), drop=True)
            ewm_stats = pd.DataFrame({f"{prefix}{col}_ewm_{int(ewm_halflife)}": ewm})
            ewm_stats[f"{prefix}{col}_momentum_ewm_{int(ewm_halflife)}"] = shifted - ewm
            parts.append(ewm_stats)
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
    with DBConnection(db_path) as conn:
        if rebuild and table_exists(conn, target_table):
            conn.execute(f"DROP TABLE IF EXISTS {target_table}")
            latest = None
        else:
            latest = get_latest_date(conn, target_table, "game_date")

        df = load_table_cached(db_path, source_table, year, rebuild=rebuild)

        hand_query = """
            SELECT b.game_pk,
                   b.opponent_team,
                   s.pitcher_hand,
                   SUM(b.strikeouts) AS strikeouts,
                   SUM(b.plate_appearances) AS plate_appearances,
                   SUM(b.ops * b.plate_appearances) / SUM(b.plate_appearances) AS team_ops
            FROM game_level_batters_vs_starters b
            JOIN game_level_starting_pitchers s
              ON b.game_pk = s.game_pk AND b.pitcher_id = s.pitcher_id
            GROUP BY b.game_pk, b.opponent_team, s.pitcher_hand
        """
        hand_df = pd.read_sql_query(hand_query, conn)
        if not hand_df.empty:
            hand_df["team_k_rate"] = (
                hand_df["strikeouts"] / hand_df["plate_appearances"]
            )
            ops_pivot = (
                hand_df.pivot(
                    index=["game_pk", "opponent_team"],
                    columns="pitcher_hand",
                    values="team_ops",
                )
                .rename(columns={"L": "team_ops_vs_LHP", "R": "team_ops_vs_RHP"})
                .reset_index()
            )
            hand_df = hand_df[
                ["game_pk", "opponent_team", "pitcher_hand", "team_k_rate"]
            ]
            df = safe_merge(
                df,
                hand_df,
                on=["game_pk", "opponent_team", "pitcher_hand"],
                how="left",
            )
            df = safe_merge(
                df,
                ops_pivot,
                on=["game_pk", "opponent_team"],
                how="left",
            )


        if "pitcher_hand" in df.columns and "bat_ops" in df.columns:
            df["bat_ops_vs_LHP"] = np.where(
                df["pitcher_hand"] == "L", df["bat_ops"], np.nan
            )
            df["bat_ops_vs_RHP"] = np.where(
                df["pitcher_hand"] == "R", df["bat_ops"], np.nan
            )
        if "pitcher_hand" in df.columns and "bat_strikeout_rate" in df.columns:
            df["bat_k_rate_vs_LHP"] = np.where(
                df["pitcher_hand"] == "L", df["bat_strikeout_rate"], np.nan
            )
            df["bat_k_rate_vs_RHP"] = np.where(
                df["pitcher_hand"] == "R", df["bat_strikeout_rate"], np.nan
            )

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
            numeric_cols=StrikeoutModelConfig.CONTEXT_ROLLING_COLS,
            ewm_halflife=StrikeoutModelConfig.EWM_HALFLIFE,
        )
        hand_cols = [
            c
            for c in [
                "team_k_rate",
                "bat_ops_vs_LHP",
                "bat_ops_vs_RHP",
                "bat_k_rate_vs_LHP",
                "bat_k_rate_vs_RHP",
            ]
            if c in df.columns
        ]
        if hand_cols:
            df = _add_group_rolling(
                df,
                ["opponent_team", "pitcher_hand"],
                "game_date",
                prefix="team_hand_",
                n_jobs=n_jobs,
                numeric_cols=hand_cols,
                ewm_halflife=StrikeoutModelConfig.EWM_HALFLIFE,
            )
            df = df.drop(columns=hand_cols)
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
    with DBConnection(db_path) as conn:
        if rebuild and table_exists(conn, target_table):
            conn.execute(f"DROP TABLE IF EXISTS {target_table}")
            latest = None
        else:
            latest = get_latest_date(conn, target_table, "game_date")

        df = load_table_cached(db_path, source_table, year, rebuild=rebuild)

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
        if "humidity" in df.columns:
            df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")

        df["day_of_week"] = df["game_date"].dt.dayofweek

        def _compute_distance(row: pd.Series) -> float:
            if "pitching_team" in row and row["pitching_team"] != row["home_team"]:
                away = row["pitching_team"]
            else:
                away = row.get("opponent_team")
            home_stadium = TEAM_TO_BALLPARK.get(row["home_team"])
            away_stadium = TEAM_TO_BALLPARK.get(away)
            home_coord = BALLPARK_COORDS.get(home_stadium)
            away_coord = BALLPARK_COORDS.get(away_stadium)
            return _haversine_distance(away_coord, home_coord)

        df["travel_distance"] = df.apply(_compute_distance, axis=1)

        df["stadium"] = df["home_team"].map(TEAM_TO_BALLPARK)
        df["park_factor"] = df["stadium"].map(BALLPARK_FACTORS)

        df = _add_group_rolling(
            df,
            ["hp_umpire"],
            "game_date",
            prefix="ump_",
            n_jobs=n_jobs,
            numeric_cols=StrikeoutModelConfig.CONTEXT_ROLLING_COLS,
            ewm_halflife=StrikeoutModelConfig.EWM_HALFLIFE,
        )
        if "weather" in df.columns:
            df = _add_group_rolling(
                df,
                ["weather"],
                "game_date",
                prefix="wx_",
                n_jobs=n_jobs,
                numeric_cols=StrikeoutModelConfig.CONTEXT_ROLLING_COLS,
                ewm_halflife=StrikeoutModelConfig.EWM_HALFLIFE,
            )
        df = _add_group_rolling(
            df,
            ["home_team"],
            "game_date",
            prefix="venue_",
            n_jobs=n_jobs,
            numeric_cols=StrikeoutModelConfig.CONTEXT_ROLLING_COLS,
            ewm_halflife=StrikeoutModelConfig.EWM_HALFLIFE,
        )

        if rebuild or not table_exists(conn, target_table):
            df.to_sql(target_table, conn, if_exists="replace", index=False)
        else:
            df.to_sql(target_table, conn, if_exists="append", index=False)
        logger.info("Saved contextual features to %s", target_table)
        return df


def engineer_lineup_trends(
    db_path: str | None = None,
    source_table: str = "game_starting_lineups",
    target_table: str = "lineup_trends",
    n_jobs: int | None = None,
    year: int | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Compute rolling lineup statistics grouped by pitcher."""

    db_path = db_path or DBConfig.PATH
    logger.info("Loading lineup data from %s", source_table)
    with DBConnection(db_path) as conn:
        if rebuild and table_exists(conn, target_table):
            conn.execute(f"DROP TABLE IF EXISTS {target_table}")
            latest = None
        else:
            latest = get_latest_date(conn, target_table, "game_date")

        df = load_table_cached(db_path, source_table, year, rebuild=rebuild)

        if "game_date" not in df.columns:
            date_df = load_table_cached(
                db_path,
                "game_level_starting_pitchers",
                year,
                rebuild=rebuild,
            )[["game_pk", "pitcher_id", "game_date"]]
            merge_cols=["game_pk"]
            if "pitcher_id" in df.columns:
                merge_cols.append("pitcher_id")
            df = safe_merge(df, date_df, on=merge_cols, how="left")
    if df.empty:
        logger.warning("No data found in %s", source_table)
        return df

    df["game_date"] = pd.to_datetime(df["game_date"])
    if latest is not None:
        df = df[df["game_date"] > latest]
    if df.empty:
        logger.info("No new rows to process for %s", target_table)
        return df

    numeric_cols = [
        c
        for c in df.select_dtypes(include=np.number).columns
        if c not in {"game_pk", "pitcher_id"}
    ]
    df = _add_group_rolling(
        df,
        ["pitcher_id"],
        "game_date",
        prefix="lineup_",
        n_jobs=n_jobs,
        numeric_cols=numeric_cols,
        ewm_halflife=StrikeoutModelConfig.EWM_HALFLIFE,
    )

    with DBConnection(db_path) as conn:
        if rebuild or not table_exists(conn, target_table):
            df.to_sql(target_table, conn, if_exists="replace", index=False)
        else:
            df.to_sql(target_table, conn, if_exists="append", index=False)
    logger.info("Saved lineup trends to %s", target_table)
    return df


def engineer_catcher_defense(
    db_path: str | None = None,
    lineup_table: str = "game_starting_lineups",
    metrics_table: str = "catcher_defense_metrics",
    target_table: str = "rolling_catcher_defense",
    n_jobs: int | None = None,
    year: int | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Merge catcher defense metrics and compute rolling stats by catcher."""

    db_path = db_path or DBConfig.PATH
    logger.info("Loading catcher data from %s", metrics_table)
    with DBConnection(db_path) as conn:
        if rebuild and table_exists(conn, target_table):
            conn.execute(f"DROP TABLE IF EXISTS {target_table}")
            latest = None
        else:
            latest = get_latest_date(conn, target_table, "game_date")

        # "catcher_id" was added in later versions of the lineup table. Infer it
        # from pitch-level data if the column is missing so the feature
        # engineering script can run on older databases.
        table_cols = [row[1] for row in conn.execute(f"PRAGMA table_info({lineup_table})")]
        select_cols = ["game_pk", "team"]
        if "catcher_id" in table_cols:
            select_cols.append("catcher_id")
        lineup_df = pd.read_sql_query(
            f"SELECT {', '.join(select_cols)} FROM {lineup_table}", conn
        )
        if "catcher_id" not in lineup_df.columns:
            logger.warning(
                "catcher_id column missing from %s; inferring from statcast_pitchers",
                lineup_table,
            )
            catcher_df = pd.read_sql_query(
                """
                SELECT game_pk, inning_topbot, home_team, away_team,
                       fielder_2, pitch_number, inning
                FROM statcast_pitchers
                WHERE inning = 1
                """,
                conn,
            )
            if catcher_df.empty:
                lineup_df["catcher_id"] = np.nan
            else:
                catcher_df["team"] = np.where(
                    catcher_df["inning_topbot"] == "Top",
                    catcher_df["home_team"],
                    catcher_df["away_team"],
                )
                catcher_map = (
                    catcher_df.sort_values("pitch_number")
                    .groupby(["game_pk", "team"], as_index=False)["fielder_2"]
                    .first()
                    .rename(columns={"fielder_2": "catcher_id"})
                )
                lineup_df = safe_merge(
                    lineup_df,
                    catcher_map,
                    on=["game_pk", "team"],
                    how="left",
                )
        date_df = pd.read_sql_query(
            "SELECT game_pk, pitching_team, pitcher_id, game_date FROM game_level_starting_pitchers",
            conn,
        )
        lineup_df = safe_merge(
            lineup_df,
            date_df,
            left_on=["game_pk", "team"],
            right_on=["game_pk", "pitching_team"],
            how="left",
        )

        metrics_query = f"SELECT * FROM {metrics_table}"
        if year:
            metrics_query += f" WHERE strftime('%Y', game_date) = '{year}'"
        metrics_df = pd.read_sql_query(metrics_query, conn)

    if metrics_df.empty or lineup_df.empty:
        logger.warning("No catcher data found")
        return pd.DataFrame()

    metrics_df["game_date"] = pd.to_datetime(metrics_df["game_date"])
    lineup_df["game_date"] = pd.to_datetime(lineup_df["game_date"])
    df = safe_merge(
        lineup_df,
        metrics_df,
        on=["game_pk", "catcher_id"],
        how="left",
        suffixes=("", "_metrics"),
    )
    if "game_date_metrics" in df.columns:
        df["game_date"] = df["game_date"].fillna(df.pop("game_date_metrics"))
    if latest is not None:
        df = df[df["game_date"] > latest]
    if df.empty:
        logger.info("No new rows to process for %s", target_table)
        return df

    numeric_cols = [
        c
        for c in metrics_df.select_dtypes(include=np.number).columns
        if c not in {"game_pk", "catcher_id"}
    ]
    df = _add_group_rolling(
        df,
        ["catcher_id"],
        "game_date",
        prefix="catcher_",
        n_jobs=n_jobs,
        numeric_cols=numeric_cols,
        ewm_halflife=StrikeoutModelConfig.EWM_HALFLIFE,
    )

    with DBConnection(db_path) as conn:
        if rebuild or not table_exists(conn, target_table):
            df.to_sql(target_table, conn, if_exists="replace", index=False)
        else:
            df.to_sql(target_table, conn, if_exists="append", index=False)
    logger.info("Saved catcher defense trends to %s", target_table)
    return df
