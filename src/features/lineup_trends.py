from __future__ import annotations

import pandas as pd
from pathlib import Path


from src.utils import (
    DBConnection,
    setup_logger,
    table_exists,
    get_latest_date,
    safe_merge,
)
from src.config import DBConfig, LogConfig, StrikeoutModelConfig
from .contextual import _add_group_rolling

logger = setup_logger(
    "lineup_trends",
    LogConfig.LOG_DIR / "lineup_trends.log",
)


def engineer_lineup_trends(
    db_path: Path | None = None,
    source_table: str = "game_starting_lineups",
    target_table: str = "rolling_lineup_features",
    n_jobs: int | None = None,
    year: int | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Aggregate starting lineup stats and compute rolling trends.

    The source table should contain one row per game, team and batting order slot
    with numeric batting metrics. Rolling averages and exponentially weighted
    means are calculated per team/slot group.
    """

    db_path = db_path or DBConfig.PATH
    logger.info("Loading starting lineup data from %s", source_table)
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
        ["opponent_team", "batting_order"],
        "game_date",
        prefix="lineup_",
        n_jobs=n_jobs,
        numeric_cols=StrikeoutModelConfig.LINEUP_ROLLING_COLS,
        ewm_halflife=StrikeoutModelConfig.EWM_HALFLIFE,
    )

    key_cols = ["game_pk", "opponent_team", "game_date"]
    projected_df = None
    weighted_df = None
    projected_cols = [c for c in df.columns if c.startswith("projected_")]
    weight_col = None
    for cand in ("projected_pa", "projected_plate_appearances"):
        if cand in projected_cols:
            weight_col = cand
            break
    if projected_cols:
        # Compute simple means for all projected metrics
        mean_df = df.groupby(key_cols)[projected_cols].mean().reset_index()
        rename_map = {
            c: f"projected_lineup_{c[len('projected_'):]}" for c in projected_cols
        }
        projected_df = mean_df.rename(columns=rename_map)

    if weight_col:
        metric_cols = [c for c in projected_cols if c != weight_col]
        if metric_cols:
            weighted = df[key_cols + metric_cols + [weight_col]].copy()
            for col in metric_cols:
                weighted[col] = weighted[col] * weighted[weight_col]
            grouped = weighted.groupby(key_cols)
            sums = grouped[metric_cols + [weight_col]].sum()
            for col in metric_cols:
                sums[col] = sums[col] / sums[weight_col]
            sums = sums.drop(columns=[weight_col]).reset_index()
            rename_map = {
                col: f"projected_lineup_{col[len('projected_'):]}_pa_weighted"
                for col in metric_cols
            }
            weighted_df = sums.rename(columns=rename_map)

    # Drop raw numeric columns
    drop_cols = [c for c in StrikeoutModelConfig.LINEUP_ROLLING_COLS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    frames = []
    for slot, g in df.groupby("batting_order"):
        slot_df = g.set_index(key_cols)[
            [c for c in g.columns if c.startswith("lineup_")]
        ].add_prefix(f"slot{int(slot)}_")
        frames.append(slot_df)

    final_df = pd.concat(frames, axis=1).reset_index()
    if projected_df is not None:
        final_df = safe_merge(final_df, projected_df, on=key_cols, how="left")
    if weighted_df is not None:
        final_df = safe_merge(final_df, weighted_df, on=key_cols, how="left")

    with DBConnection(db_path) as conn:
        if rebuild or not table_exists(conn, target_table):
            final_df.to_sql(target_table, conn, index=False, if_exists="replace")
        else:
            final_df.to_sql(target_table, conn, index=False, if_exists="append")
    logger.info("Saved lineup trends to %s", target_table)
    return final_df
