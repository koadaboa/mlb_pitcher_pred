from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.utils import DBConnection, setup_logger
from src.utils import table_exists, get_latest_date
from src.config import DBConfig, LogConfig, StrikeoutModelConfig
import re

logger = setup_logger("join_features", LogConfig.LOG_DIR / "join_features.log")


def build_model_features(
    db_path: Path | None = None,
    pitcher_table: str = "rolling_pitcher_features",
    opp_table: str = "rolling_pitcher_vs_team",
    context_table: str = "contextual_features",
    target_table: str = "model_features",
    year: int | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Join engineered feature tables into one dataset.

    Parameters
    ----------
    rebuild : bool, default False
        Drop ``target_table`` and recreate it with only the configured window
        sizes.
    """
    db_path = db_path or DBConfig.PATH

    valid_windows = {str(w) for w in StrikeoutModelConfig.WINDOW_SIZES}
    pattern = re.compile(r"_(?:mean|std|momentum)_(\d+)$")

    with DBConnection(db_path) as conn:
        if rebuild and table_exists(conn, target_table):
            conn.execute(f"DROP TABLE IF EXISTS {target_table}")
            latest = None
        else:
            latest = get_latest_date(conn, target_table, "game_date")

        base_query = "SELECT * FROM {}"
        filter_clause = f" WHERE strftime('%Y', game_date) = '{year}'" if year else ""
        pitcher_df = pd.read_sql_query(
            base_query.format(pitcher_table) + filter_clause, conn
        )
        opp_df = pd.read_sql_query(base_query.format(opp_table) + filter_clause, conn)
        ctx_df = pd.read_sql_query(
            base_query.format(context_table) + filter_clause, conn
        )

        if pitcher_df.empty:
            logger.warning("No data found in %s", pitcher_table)
            return pitcher_df

        df = pitcher_df.merge(opp_df, on=["game_pk", "pitcher_id"], how="left")
        df = df.merge(ctx_df, on=["game_pk", "pitcher_id"], how="left")

        drop_ump_cols = [
            c
            for c in df.columns
            if c.startswith("1b_umpire") or c.startswith("2b_umpire") or c.startswith("3b_umpire")
        ]
        if drop_ump_cols:
            df = df.drop(columns=drop_ump_cols)
        if latest is not None:
            df = df[df["game_date"] > latest]

        # Drop columns created with window sizes not in config
        drop_cols = [
            c
            for c in df.columns
            if (m := pattern.search(c)) and m.group(1) not in valid_windows
        ]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        if df.empty:
            logger.info("No new rows to process for %s", target_table)
            return df

        if rebuild or not table_exists(conn, target_table):
            df.to_sql(target_table, conn, if_exists="replace", index=False)
        else:
            df.to_sql(target_table, conn, if_exists="append", index=False)
        logger.info("Saved joined features to %s", target_table)
        return df
