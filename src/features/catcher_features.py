import pandas as pd
import numpy as np
import logging
from typing import Callable, List, Dict, Tuple

logger = logging.getLogger(__name__)


def calculate_catcher_rolling_features(
    catcher_hist_df: pd.DataFrame,
    group_col: str,
    date_col: str,
    metrics: List[str],
    windows: List[int],
    min_periods: int,
    calculate_multi_window_rolling: Callable,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Calculate rolling framing metrics for catchers."""
    if catcher_hist_df is None or catcher_hist_df.empty:
        logger.warning("Input DataFrame for catcher rolling features is empty.")
        return pd.DataFrame(), {}

    available_metrics = [m for m in metrics if m in catcher_hist_df.columns]
    if not available_metrics:
        logger.warning("No specified catcher metrics found in DataFrame.")
        return pd.DataFrame(index=catcher_hist_df.index), {}

    rolling_calc = calculate_multi_window_rolling(
        df=catcher_hist_df,
        group_col=group_col,
        date_col=date_col,
        metrics=available_metrics,
        windows=windows,
        min_periods=min_periods,
    )

    rename_map = {
        f"{m}_roll{w}g": f"c_roll{w}g_{m}"
        for w in windows
        for m in available_metrics
        if f"{m}_roll{w}g" in rolling_calc.columns
    }
    rolling_df = rolling_calc.rename(columns=rename_map)
    rolling_df[group_col] = catcher_hist_df[group_col]
    rolling_df[date_col] = catcher_hist_df[date_col]
    logger.info(
        f"Finished calculating catcher rolling features. Found {len(rename_map)} features."
    )
    return rolling_df, rename_map


def merge_catcher_features_historical(
    final_features_df: pd.DataFrame,
    catcher_rolling_df: pd.DataFrame,
    catcher_rename_map: Dict[str, str],
) -> pd.DataFrame:
    """Merge historical catcher rolling features using merge_asof."""
    if catcher_rolling_df is None or catcher_rolling_df.empty:
        logger.warning("Catcher rolling features DataFrame is empty, skipping merge.")
        return final_features_df
    if 'catcher_id' not in final_features_df.columns:
        logger.error("'catcher_id' column missing from features for catcher merge.")
        return final_features_df

    cols_to_merge = list(catcher_rename_map.values())
    cols_to_merge = [c for c in cols_to_merge if c in catcher_rolling_df.columns]
    if not cols_to_merge:
        logger.warning("No catcher rolling columns identified to merge.")
        return final_features_df

    final_sorted = final_features_df.sort_values('game_date').copy()
    catcher_sorted = catcher_rolling_df[[group_col:= 'catcher_id', 'game_date'] + cols_to_merge].sort_values('game_date')
    final_sorted['merge_key_catcher'] = final_sorted['catcher_id'].astype(str)
    catcher_sorted['merge_key_catcher'] = catcher_sorted['catcher_id'].astype(str)

    merged = pd.merge_asof(
        final_sorted,
        catcher_sorted.drop(columns=['catcher_id']),
        on='game_date',
        left_by='merge_key_catcher',
        right_by='merge_key_catcher',
        direction='backward',
        allow_exact_matches=False,
    ).drop(columns=['merge_key_catcher'])
    return merged


def merge_catcher_features_prediction(
    final_features_df: pd.DataFrame,
    latest_catcher_rolling: pd.DataFrame,
    catcher_rename_map: Dict[str, str],
) -> pd.DataFrame:
    """Merge latest catcher rolling stats onto prediction baseline."""
    if latest_catcher_rolling is None or latest_catcher_rolling.empty:
        logger.warning("Latest catcher rolling DataFrame is empty, skipping merge.")
        return final_features_df
    if 'catcher_id' not in final_features_df.columns:
        logger.error("'catcher_id' missing from prediction baseline for merge.")
        return final_features_df

    cols_to_merge = ['catcher_id'] + list(catcher_rename_map.values())
    cols_to_merge = [c for c in cols_to_merge if c in latest_catcher_rolling.columns]
    if len(cols_to_merge) <= 1:
        logger.warning("No catcher rolling columns to merge for prediction.")
        return final_features_df

    merged = pd.merge(
        final_features_df,
        latest_catcher_rolling[cols_to_merge],
        on='catcher_id',
        how='left',
    )
    return merged
