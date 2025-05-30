from __future__ import annotations

import pandas as pd
from pathlib import Path


from src.utils import DBConnection, setup_logger, table_exists, get_latest_date
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

    # Drop raw numeric columns
    drop_cols = [c for c in StrikeoutModelConfig.LINEUP_ROLLING_COLS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    key_cols = ["game_pk", "opponent_team", "game_date"]
    frames = []
    for slot, g in df.groupby("batting_order"):
        slot_df = g.set_index(key_cols)[
            [c for c in g.columns if c.startswith("lineup_")]
        ].add_prefix(f"slot{int(slot)}_")
        frames.append(slot_df)

    final_df = pd.concat(frames, axis=1).reset_index()

    with DBConnection(db_path) as conn:
        if rebuild or not table_exists(conn, target_table):
            final_df.to_sql(target_table, conn, index=False, if_exists="replace")
        else:
            final_df.to_sql(target_table, conn, index=False, if_exists="append")
    logger.info("Saved lineup trends to %s", target_table)
    return final_df

