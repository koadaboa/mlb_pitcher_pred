from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.utils import DBConnection, setup_logger
from src.config import DBConfig, LogConfig

logger = setup_logger("join_features", LogConfig.LOG_DIR / "join_features.log")


def build_model_features(
    db_path: Path | None = None,
    pitcher_table: str = "rolling_pitcher_features",
    opp_table: str = "rolling_pitcher_vs_team",
    context_table: str = "contextual_features",
    target_table: str = "model_features",
) -> pd.DataFrame:
    """Join engineered feature tables into one dataset."""
    db_path = db_path or DBConfig.PATH

    with DBConnection(db_path) as conn:
        pitcher_df = pd.read_sql_query(f"SELECT * FROM {pitcher_table}", conn)
        opp_df = pd.read_sql_query(f"SELECT * FROM {opp_table}", conn)
        ctx_df = pd.read_sql_query(f"SELECT * FROM {context_table}", conn)

    if pitcher_df.empty:
        logger.warning("No data found in %s", pitcher_table)
        return pitcher_df

    df = pitcher_df.merge(opp_df, on=["game_pk", "pitcher_id"], how="left")
    df = df.merge(ctx_df, on=["game_pk", "pitcher_id"], how="left")

    with DBConnection(db_path) as conn:
        df.to_sql(target_table, conn, if_exists="replace", index=False)
    logger.info("Saved joined features to %s", target_table)
    return df
