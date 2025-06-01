from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.utils import DBConnection, setup_logger, table_exists, get_latest_date
from src.config import DBConfig, LogConfig, StrikeoutModelConfig
from .engineer_features import add_rolling_features

logger = setup_logger(
    "batter_pitcher_history",
    LogConfig.LOG_DIR / "batter_pitcher_history.log",
)


def engineer_batter_pitcher_history(
    db_path: Path | None = None,
    source_table: str = "game_level_batters_vs_starters",
    date_table: str = "game_level_starting_pitchers",
    target_table: str = "rolling_batter_pitcher_history",
    year: int | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Compute rolling batter vs pitcher history metrics."""

    db_path = db_path or DBConfig.PATH
    logger.info("Loading batter vs starter data from %s", source_table)
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

        if "game_date" not in df.columns:
            date_df = pd.read_sql_query(
                f"SELECT game_pk, game_date FROM {date_table}", conn
            )
            df = df.merge(date_df, on="game_pk", how="left")

    if df.empty:
        logger.warning("No data found in %s", source_table)
        return df

    df["game_date"] = pd.to_datetime(df["game_date"])
    if latest is not None:
        df = df[df["game_date"] > latest]
    if df.empty:
        logger.info("No new rows to process for %s", target_table)
        return df

    group_cols = ["pitcher_id", "batter_id", "game_date"]
    agg_dict = {
        "game_pk": "first",
        "plate_appearances": "sum",
        "strikeouts": "sum",
        "ops": "mean",
    }
    if "swings" in df.columns:
        agg_dict["swings"] = "sum"
    if "whiffs" in df.columns:
        agg_dict["whiffs"] = "sum"

    agg = df.groupby(group_cols).agg(agg_dict).reset_index()

    agg["batter_so_rate"] = agg["strikeouts"] / agg["plate_appearances"]
    if "swings" in df.columns and "whiffs" in df.columns:
        agg["batter_whiff_rate"] = agg["whiffs"] / agg["swings"]
    else:
        agg["batter_whiff_rate"] = pd.NA
    agg["batter_ops"] = agg["ops"]

    pair_key = agg["pitcher_id"].astype(str) + "_" + agg["batter_id"].astype(str)
    agg["pair_key"] = pair_key

    agg = add_rolling_features(
        agg,
        group_col="pair_key",
        date_col="game_date",
        windows=StrikeoutModelConfig.WINDOW_SIZES,
        numeric_cols=["batter_so_rate", "batter_ops", "batter_whiff_rate"],
        ewm_halflife=StrikeoutModelConfig.EWM_HALFLIFE,
    )

    agg = agg.drop(columns=["pair_key"])

    with DBConnection(db_path) as conn:
        if rebuild or not table_exists(conn, target_table):
            agg.to_sql(target_table, conn, if_exists="replace", index=False)
        else:
            agg.to_sql(target_table, conn, if_exists="append", index=False)
    logger.info("Saved batter-pitcher history to %s", target_table)
    return agg


if __name__ == "__main__":
    engineer_batter_pitcher_history()
