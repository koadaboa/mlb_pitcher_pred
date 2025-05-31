from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.utils import DBConnection, setup_logger
from src.utils import table_exists, get_latest_date
from src.config import DBConfig, LogConfig, StrikeoutModelConfig
from .encoding import mean_target_encode
from .selection import BASE_EXCLUDE_COLS
import re
import numpy as np

logger = setup_logger("join_features", LogConfig.LOG_DIR / "join_features.log")


def _winsorize_columns(df: pd.DataFrame, cols: list[str], lower_q: float = 0.01, upper_q: float = 0.99) -> None:
    """Clip numeric columns to specified quantiles to limit outliers."""
    for col in cols:
        lower = df[col].quantile(lower_q)
        upper = df[col].quantile(upper_q)
        df[col] = df[col].clip(lower, upper)


def _log_transform(df: pd.DataFrame, cols: list[str]) -> None:
    """Apply log1p transform to positive-valued columns and store as new features."""
    for col in cols:
        if (df[col] >= 0).all() and df[col].max() > 1:
            df[f"log_{col}"] = np.log1p(df[col])


def build_model_features(
    db_path: Path | None = None,
    pitcher_table: str = "rolling_pitcher_features",
    opp_table: str = "rolling_pitcher_vs_team",
    context_table: str = "contextual_features",
    lineup_table: str = "lineup_trends",
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
        lineup_df = pd.read_sql_query(
            base_query.format(lineup_table) + filter_clause, conn
        )

        # ``game_date`` should be identical across tables for the same ``game_pk``
        # and ``pitcher_id``. Drop duplicate instances from the right-hand
        # DataFrames before merging to avoid pandas adding suffixes that can clash
        # on subsequent merges.
        for frame in (opp_df, ctx_df, lineup_df):
            if "game_date" in frame.columns:
                frame.drop(columns=["game_date"], inplace=True)

        if pitcher_df.empty:
            logger.warning("No data found in %s", pitcher_table)
            return pitcher_df

        df = pitcher_df.merge(opp_df, on=["game_pk", "pitcher_id"], how="left")
        df = df.merge(ctx_df, on=["game_pk", "pitcher_id"], how="left")
        df = df.merge(lineup_df, on=["game_pk", "pitcher_id"], how="left")

        # Deduplicate any columns that were suffixed during the merges
        dup_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
        for col in dup_cols:
            if col not in df.columns:
                continue
            base = col[:-2]
            alt = base + ("_y" if col.endswith("_x") else "_x")
            if base not in df.columns:
                if alt in df.columns:
                    if df[col].equals(df[alt]):
                        df = df.drop(columns=[alt])
                    else:
                        df[col] = df[col].combine_first(df[alt])
                        df = df.drop(columns=[alt])
                df = df.rename(columns={col: base})
            else:
                df = df.drop(columns=[col], errors="ignore")

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

        # Retain all columns after dropping unsupported window sizes. Previously
        # numeric columns that weren't rolled or explicitly listed in
        # ``ALLOWED_BASE_NUMERIC_COLS`` were discarded.  Keeping them allows the
        # model to consider raw game statistics as well.
        target = StrikeoutModelConfig.TARGET_VARIABLE
        keep_cols = list(df.columns)
        df = df[keep_cols]

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
        _winsorize_columns(df, numeric_cols)
        _log_transform(df, numeric_cols)
        cat_cols = [
            c
            for c in df.columns
            if c not in BASE_EXCLUDE_COLS
            and not pd.api.types.is_numeric_dtype(df[c])
            and c != target
        ]
        if cat_cols:
            df, _ = mean_target_encode(df, cat_cols, target)
        if df.empty:
            logger.info("No new rows to process for %s", target_table)
            return df

        if rebuild or not table_exists(conn, target_table):
            df.to_sql(target_table, conn, if_exists="replace", index=False)
        else:
            df.to_sql(target_table, conn, if_exists="append", index=False)
        logger.info("Saved joined features to %s", target_table)
        return df
