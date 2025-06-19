from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.utils import (
    DBConnection,
    setup_logger,
    table_exists,
    get_latest_date,
    safe_merge,
    load_table_cached,
)
from src.config import DBConfig, LogConfig, StrikeoutModelConfig
from .encoding import mean_target_encode
from .selection import BASE_EXCLUDE_COLS

# Categorical columns that should be ignored when mean target encoding
EXTRA_CAT_EXCLUDE_COLS = [
    "away_pitcher_ids",
    "home_pitcher_ids",
    "away_starting_pitcher_id",
    "home_starting_pitcher_id",
    "scraped_timestamp",
]
import re
import numpy as np

# Maximum number of columns supported by SQLite
SQLITE_MAX_COLUMNS = 2000

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
    catcher_table: str = "rolling_catcher_defense",
    batter_history_table: str = "rolling_batter_pitcher_history",
    lineup_ids_table: str = "game_starting_lineups",

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

        pitcher_df = load_table_cached(db_path, pitcher_table, year, rebuild=rebuild)
        opp_df = load_table_cached(db_path, opp_table, year, rebuild=rebuild)
        ctx_df = load_table_cached(db_path, context_table, year, rebuild=rebuild)
        lineup_df = load_table_cached(db_path, lineup_table, year, rebuild=rebuild)
        catcher_df = load_table_cached(db_path, catcher_table, year, rebuild=rebuild)
        bp_df = load_table_cached(db_path, batter_history_table, year, rebuild=rebuild)
        lineup_ids = load_table_cached(db_path, lineup_ids_table, rebuild=rebuild)

        # ``game_date`` should be identical across tables for the same ``game_pk``
        # and ``pitcher_id``. Drop duplicate instances from the right-hand
        # DataFrames before merging to avoid pandas adding suffixes that can clash
        # on subsequent merges.
        for frame in (opp_df, ctx_df, lineup_df, catcher_df, bp_df, lineup_ids):

            if "game_date" in frame.columns:
                frame.drop(columns=["game_date"], inplace=True)

        if pitcher_df.empty:
            logger.warning("No data found in %s", pitcher_table)
            return pitcher_df

        df = safe_merge(pitcher_df, opp_df, on=["game_pk", "pitcher_id"], how="left")
        df = df.drop_duplicates(subset=["game_pk", "pitcher_id"])
        df = safe_merge(df, ctx_df, on=["game_pk", "pitcher_id"], how="left")
        df = df.drop_duplicates(subset=["game_pk", "pitcher_id"])
        df = safe_merge(df, lineup_df, on=["game_pk", "pitcher_id"], how="left")
        df = df.drop_duplicates(subset=["game_pk", "pitcher_id"])
        df = safe_merge(df, catcher_df, on=["game_pk", "pitcher_id"], how="left")
        df = df.drop_duplicates(subset=["game_pk", "pitcher_id"])

        if not bp_df.empty and not lineup_ids.empty:
            if "team" in lineup_ids.columns and "opponent_team" not in lineup_ids.columns:
                lineup_ids = lineup_ids.rename(columns={"team": "opponent_team"})
            lineup_ids = safe_merge(
                lineup_ids,
                pitcher_df[["game_pk", "pitcher_id", "opponent_team"]],
                on=["game_pk", "opponent_team"],
                how="inner",
            )
            pair_merge = safe_merge(
                lineup_ids,
                bp_df,
                on=["game_pk", "pitcher_id", "batter_id"],
                how="left",
            )
            numeric_cols = [
                c
                for c in bp_df.columns
                if pd.api.types.is_numeric_dtype(bp_df[c])
                and c not in {"game_pk", "pitcher_id", "batter_id"}
            ]
            if numeric_cols:
                agg_df = (
                    pair_merge.groupby(["game_pk", "pitcher_id"])[numeric_cols]
                    .mean()
                    .add_prefix("opp_batter_")
                    .reset_index()
                )
                df = safe_merge(df, agg_df, on=["game_pk", "pitcher_id"], how="left")

        # Drop identifier columns that could leak future information before
        # any transformations are applied.
        df = df.drop(columns=[c for c in EXTRA_CAT_EXCLUDE_COLS if c in df.columns])

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

        # Drop numeric columns that would leak information. Only rolling stats
        # or explicitly whitelisted numeric columns are kept.
        allowed_numeric = set(StrikeoutModelConfig.ALLOWED_BASE_NUMERIC_COLS)
        target = StrikeoutModelConfig.TARGET_VARIABLE
        keep_cols = []
        for col in df.columns:
            if pattern.search(col):
                keep_cols.append(col)
            elif col in allowed_numeric or col == target:
                keep_cols.append(col)
            elif not pd.api.types.is_numeric_dtype(df[col]):
                keep_cols.append(col)
        df = df[keep_cols]

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
        _winsorize_columns(df, numeric_cols)
        _log_transform(df, numeric_cols)
        # Drop identifier columns entirely so they cannot be encoded or used as
        # raw features
        df = df.drop(columns=[c for c in EXTRA_CAT_EXCLUDE_COLS if c in df.columns])
        exclude_set = set(BASE_EXCLUDE_COLS).union(EXTRA_CAT_EXCLUDE_COLS)
        cat_cols = [
            c
            for c in df.columns
            if c not in exclude_set
            and not pd.api.types.is_numeric_dtype(df[c])
            and c != target
        ]
        if cat_cols:
            train_mask = df["game_date"].dt.year.isin(
                StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
            )
            if not train_mask.any() or not (~train_mask).any():
                df, _ = mean_target_encode(df, cat_cols, target)
            else:
                train_df, enc_map = mean_target_encode(
                    df.loc[train_mask], cat_cols, target
                )
                test_df, _ = mean_target_encode(
                    df.loc[~train_mask], cat_cols, mapping=enc_map
                )
                df = pd.concat([train_df, test_df]).sort_index()
        if df.empty:
            logger.info("No new rows to process for %s", target_table)
            return df
        col_count = len(df.columns)
        logger.info(
            "Attempting to write %d columns to %s",
            col_count,
            target_table,
        )

        if col_count > SQLITE_MAX_COLUMNS:
            logger.error(
                "SQLite supports a maximum of %d columns, but %d were provided",
                SQLITE_MAX_COLUMNS,
                col_count,
            )
            raise ValueError(
                f"Too many columns ({col_count}) for SQLite table {target_table}"
            )

        if rebuild or not table_exists(conn, target_table):
            df.to_sql(target_table, conn, if_exists="replace", index=False)
        else:
            df.to_sql(target_table, conn, if_exists="append", index=False)
        logger.info("Saved joined features to %s", target_table)
        return df
