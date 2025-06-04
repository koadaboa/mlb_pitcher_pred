from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils import DBConnection, setup_logger, table_exists
from src.config import DBConfig, LogConfig

logger = setup_logger(
    "create_catcher_defense",
    LogConfig.LOG_DIR / "create_catcher_defense.log",
)


def _compute_metrics(df: pd.DataFrame) -> Dict:
    """Compute basic framing metrics for one catcher/game."""
    taken = ~df["description"].str.contains("swing", case=False, na=False)
    called = df["description"].eq("called_strike")
    called_strike_rate = called[taken].mean() if taken.any() else np.nan
    framing_runs = called.sum()  # placeholder for more advanced metric
    below_zone_rate = np.nan
    if "plate_z" in df.columns:
        called_total = called.sum()
        if called_total:
            below_zone = (called & (df["plate_z"] < 1.5)).sum()
            below_zone_rate = below_zone / called_total
    first = df.iloc[0]
    return {
        "game_pk": first["game_pk"],
        "game_date": first["game_date"],
        "catcher_id": first["fielder_2"],
        "called_strike_rate": called_strike_rate,
        "framing_runs": framing_runs,
        "below_zone_called_strike_rate": below_zone_rate,
    }


def build_catcher_defense_metrics(
    db_path: Path = DBConfig.PATH,
    source_table: str = "statcast_pitchers",
    target_table: str = "catcher_defense_metrics",
    rebuild: bool = False,
) -> pd.DataFrame:
    """Aggregate pitch-level data into per game catcher framing metrics."""
    with DBConnection(db_path) as conn:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({source_table})")]
        select_cols = ["game_pk", "game_date", "fielder_2", "description"]
        if "plate_z" in cols:
            select_cols.append("plate_z")
        df = pd.read_sql_query(
            f"SELECT {', '.join(select_cols)} FROM {source_table}",
            conn,
        )
    if df.empty:
        logger.warning("No data found in %s", source_table)
        return df

    df = df.dropna(subset=["fielder_2"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    rows: List[Dict] = []
    for _, g in df.groupby(["game_pk", "fielder_2", "game_date"], sort=False):
        rows.append(_compute_metrics(g))
    result = pd.DataFrame(rows)

    with DBConnection(db_path) as conn:
        if rebuild or not table_exists(conn, target_table):
            result.to_sql(target_table, conn, index=False, if_exists="replace")
        else:
            result.to_sql(target_table, conn, index=False, if_exists="append")
    logger.info("Saved catcher defense metrics to %s", target_table)
    return result


if __name__ == "__main__":
    try:
        build_catcher_defense_metrics()
    except Exception as exc:
        logger.exception("Failed to build catcher defense metrics: %s", exc)
