import logging
from collections import defaultdict
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DBConnection, setup_logger
try:
    from src.config import DBConfig, LogConfig
except Exception:  # pragma: no cover - fallback for standalone execution
    class DBConfig:
        PATH = Path("data/pitcher_stats.db")
    class LogConfig:
        LOG_DIR = Path("logs")
        LOG_DIR.mkdir(parents=True, exist_ok=True)

STARTERS_TABLE = "game_level_starting_pitchers"
PITCHERS_TABLE = "statcast_pitchers"
CHUNK_SIZE = 500_000

logger = setup_logger(
    "create_starting_pitcher_table",
    LogConfig.LOG_DIR / "create_starting_pitcher_table.log",
)

def get_candidate_starters(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return DataFrame of first pitchers appearing for each team in inning 1."""
    query = f"""
    WITH pitch_team AS (
        SELECT
            game_pk,
            game_date,
            CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END AS pitching_team,
            CASE WHEN inning_topbot = 'Top' THEN away_team ELSE home_team END AS opponent_team,
            pitcher AS pitcher_id,
            inning,
            at_bat_number,
            pitch_number
        FROM {PITCHERS_TABLE}
    ), ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY game_pk, pitching_team
                ORDER BY inning, at_bat_number, pitch_number
            ) AS rn
        FROM pitch_team
    )
    SELECT game_pk, game_date, pitching_team, opponent_team, pitcher_id
    FROM ranked
    WHERE rn = 1 AND inning = 1
    """
    return pd.read_sql_query(query, conn)


def load_pitch_data(conn: sqlite3.Connection, starters: pd.DataFrame) -> pd.DataFrame:
    """Load statcast data only for candidate starters using chunked reads."""
    starter_keys = set(zip(starters.game_pk, starters.pitcher_id))
    cols = [
        "game_pk",
        "game_date",
        "pitcher",
        "release_speed",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "plate_z",
        "release_pos_x",
        "release_pos_y",
        "release_pos_z",
        "release_extension",
        "sz_top",
        "sz_bot",
        "spin_axis",
        "release_spin_rate",
        "effective_speed",
        "n_thruorder_pitcher",
        "pitcher_days_since_prev_game",
        "balls",
        "strikes",
        "outs_when_up",
        "inning",
        "pitch_type",
        "description",
        "p_throws",
        "stand",
    ]
    query = f"SELECT {', '.join(cols)} FROM {PITCHERS_TABLE}"
    chunks = []
    for chunk in pd.read_sql_query(query, conn, chunksize=CHUNK_SIZE):
        keys = list(zip(chunk.game_pk, chunk.pitcher))
        mask = [k in starter_keys for k in keys]
        filtered = chunk.loc[mask]
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame(columns=cols)
    return pd.concat(chunks, ignore_index=True)


def aggregate_starting_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level data to game-level starting pitcher stats."""
    if df.empty:
        return df

    df = df.copy()
    df["handedness_matchup"] = (
        df["p_throws"].str.upper().str[0] + "_vs_" + df["stand"].str.upper().str[0]
    )

    group_cols = ["game_date", "pitcher"]

    agg_map = {
        "release_speed": ["mean", "std", "min", "max"],
        "pfx_x": ["mean", "std", "min", "max"],
        "pfx_z": ["mean", "std", "min", "max"],
        "plate_x": ["mean", "std"],
        "plate_z": ["mean", "std"],
        "release_pos_x": ["mean", "std"],
        "release_pos_z": ["mean", "std"],
        "release_pos_y": ["mean", "std"],
        "release_extension": ["mean", "std"],
        "sz_top": ["mean"],
        "sz_bot": ["mean"],
        "spin_axis": ["mean", "std"],
        "release_spin_rate": ["mean", "std"],
        "effective_speed": ["mean", "std"],
        "n_thruorder_pitcher": ["max"],
        "pitcher_days_since_prev_game": ["mean"],
        "balls": ["mean"],
        "strikes": ["mean"],
        "outs_when_up": ["mean"],
        "inning": ["max"],
    }

    agg_df = df.groupby(group_cols).agg(agg_map)
    agg_df.columns = ["_".join(col) for col in agg_df.columns]

    agg_df["total_pitches"] = df.groupby(group_cols).size().astype(int)

    # pitch type distribution
    pitch_dist = (
        df.groupby(group_cols)["pitch_type"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    pitch_dist.columns = [f"pitch_type_{c}" for c in pitch_dist.columns]
    agg_df = agg_df.join(pitch_dist, how="left")

    # whiff rate
    whiff_rate = (
        df.groupby(group_cols)["description"]
        .apply(lambda x: (x == "swinging_strike").mean())
        .rename("whiff_rate")
    )
    agg_df = agg_df.join(whiff_rate, how="left")

    # handedness matchup distribution
    hm_dist = (
        df.groupby(group_cols)["handedness_matchup"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    hm_dist.columns = [f"matchup_{c}" for c in hm_dist.columns]
    agg_df = agg_df.join(hm_dist, how="left")

    agg_df = agg_df.reset_index()
    return agg_df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop high-NaN columns, log missingness, and impute remaining values."""
    if df.empty:
        return df

    nan_pct = df.isna().mean()

    try:
        nan_pct.to_csv("nan_percentages.csv")
    except Exception as exc:  # pragma: no cover - logging only
        logger.warning("Could not write nan_percentages.csv: %s", exc)

    nan_pct[(nan_pct >= 0.15) & (nan_pct <= 0.5)].to_csv(
        "nan_log_starting_pitchers.csv"
    )


    drop_cols = nan_pct[nan_pct > 0.25].index.tolist()
    if drop_cols:
        logger.info("Dropping columns due to missingness: %s", drop_cols)
        df = df.drop(columns=drop_cols)

    # Impute remaining
    count_like = [
        c
        for c in df.columns
        if any(
            c.startswith(prefix)
            for prefix in [
                "n_thruorder_pitcher",
                "balls",
                "strikes",
                "outs_when_up",
                "total_pitches",
            ]
        )
    ]
    for col in df.columns:
        if col in count_like:
            df[col] = df[col].fillna(0)
        else:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
    return df


def main(db_path: Path = DBConfig.PATH) -> None:
    """Build or replace the aggregated starting pitcher table."""
    logger.info("Loading candidate starting pitchers")
    with DBConnection(db_path) as conn:
        starters = get_candidate_starters(conn)
        if starters.empty:
            logger.warning("No candidate starters found")
            return
        logger.info("Found %d candidate starters", len(starters))
        pitch_df = load_pitch_data(conn, starters)
        if pitch_df.empty:
            logger.warning("No pitch data loaded for starters")
            return
        agg_df = aggregate_starting_pitchers(pitch_df)

        # Filter by pitch count and optional quality metric
        agg_df = agg_df[agg_df["total_pitches"] >= 50]
        if "n_thruorder_pitcher_max" in agg_df.columns:
            agg_df = agg_df[agg_df["n_thruorder_pitcher_max"] >= 1.5]

        agg_df = agg_df.merge(
            starters,
            left_on=["game_date", "pitcher"],
            right_on=["game_date", "pitcher_id"],
            how="left",
        ).drop(columns=["pitcher_id"])

        agg_df = handle_missing_values(agg_df)
        agg_df.to_sql(STARTERS_TABLE, conn, if_exists="replace", index=False)
        logger.info("Wrote %d rows to %s", len(agg_df), STARTERS_TABLE)


if __name__ == "__main__":
    main()
