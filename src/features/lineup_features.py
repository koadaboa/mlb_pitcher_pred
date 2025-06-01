import pandas as pd
import logging
from typing import Callable, List, Dict

logger = logging.getLogger(__name__)


def calculate_batter_rolling_features(
    batter_hist_df: pd.DataFrame,
    group_col: str,
    date_col: str,
    metrics: List[str],
    windows: List[int],
    min_periods: int,
    calculate_multi_window_rolling: Callable,
) -> (pd.DataFrame, Dict[str, str]):
    """Calculate rolling stats for batters."""
    if batter_hist_df is None or batter_hist_df.empty:
        logger.warning("Batter history DataFrame is empty.")
        return pd.DataFrame(), {}

    available_metrics = [m for m in metrics if m in batter_hist_df.columns]
    if not available_metrics:
        logger.warning("No batter metrics found for rolling calculation.")
        return pd.DataFrame(), {}

    rolling_df = calculate_multi_window_rolling(
        df=batter_hist_df,
        group_col=group_col,
        date_col=date_col,
        metrics=available_metrics,
        windows=windows,
        min_periods=min_periods,
    )

    rename_map = {
        f"{m}_roll{w}g": f"b_roll{w}g_{m}"
        for w in windows
        for m in available_metrics
        if f"{m}_roll{w}g" in rolling_df.columns
    }
    rolling_df = rolling_df.rename(columns=rename_map)
    rolling_df[group_col] = batter_hist_df[group_col]
    rolling_df[date_col] = batter_hist_df[date_col]
    return rolling_df, rename_map


def aggregate_lineup_metrics(
    lineup_df: pd.DataFrame,
    batter_rolling_df: pd.DataFrame,
    windows: List[int],
) -> pd.DataFrame:
    """Aggregate batter rolling metrics across a starting lineup."""
    if lineup_df is None or lineup_df.empty or batter_rolling_df is None or batter_rolling_df.empty:
        logger.warning("Lineup or batter rolling data empty; skipping aggregation.")
        return pd.DataFrame()

    merge_cols = ["batter_id", "game_date"]
    merged = pd.merge(lineup_df, batter_rolling_df, on=merge_cols, how="left")

    grouped = merged.groupby(["game_pk", "team_abbr"])
    frames = []
    for w in windows:
        cols = [c for c in batter_rolling_df.columns if c.startswith(f"b_roll{w}g_")]
        if not cols:
            continue
        agg = grouped[cols].mean()
        rename = {}
        for c in cols:
            base = c.replace(f"b_roll{w}g_", "")
            if base.endswith("_bat"):
                base = base[:-4]
            rename[c] = f"lineup_{base}_mean_roll{w}g"
        frames.append(agg.rename(columns=rename))

    if not frames:
        logger.warning("No lineup metrics calculated.")
        return pd.DataFrame()

    return pd.concat(frames, axis=1).reset_index()
