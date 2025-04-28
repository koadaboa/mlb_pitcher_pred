# src/features/ballpark_features.py
import pandas as pd
import numpy as np
import logging
from typing import Callable, List, Dict, Tuple

logger = logging.getLogger(__name__)

def calculate_ballpark_rolling_features(
    pitcher_hist_df: pd.DataFrame, # Input uses pitcher history for ballpark metric
    group_col: str,
    date_col: str,
    metrics: List[str],
    windows: List[int],
    min_periods: int,
    calculate_multi_window_rolling: Callable
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Calculates rolling features specifically for ballparks.

    Args:
        pitcher_hist_df: DataFrame containing pitcher game-level data,
                         must include 'ballpark', 'game_date', and the metric columns.
        group_col: The column to group by (e.g., 'ballpark').
        date_col: The date column for sorting and rolling.
        metrics: List of metric columns to calculate rolling stats for (e.g., ['k_percent']).
        windows: List of window sizes.
        min_periods: Minimum number of observations in window required.
        calculate_multi_window_rolling: The function to perform the rolling calculation.

    Returns:
        A tuple containing:
            - DataFrame with ballpark rolling features, including 'ballpark' and 'game_date' keys.
              Columns are renamed with a 'bp_' prefix (e.g., 'bp_roll5g_k_percent').
            - Dictionary mapping original rolling column names to renamed ballpark column names.
              Returns empty DataFrame and empty dict if input is invalid or empty.
    """
    if pitcher_hist_df is None or pitcher_hist_df.empty:
        logger.warning("Input DataFrame for ballpark rolling features is empty.")
        return pd.DataFrame(), {}

    logger.info(f"Calculating ballpark rolling features (Windows: {windows})...")
    available_metrics = [m for m in metrics if m in pitcher_hist_df.columns]
    if not available_metrics:
        logger.warning("No specified ballpark metrics found in the input DataFrame.")
        return pd.DataFrame(index=pitcher_hist_df.index), {}
    if group_col not in pitcher_hist_df.columns:
        logger.error(f"Group column '{group_col}' not found for ballpark rolling features.")
        return pd.DataFrame(), {}

    ballpark_rolling_calc = calculate_multi_window_rolling(
        df=pitcher_hist_df,
        group_col=group_col,
        date_col=date_col,
        metrics=available_metrics,
        windows=windows,
        min_periods=min_periods
    )

    # Define and apply rename map
    rename_map = {
        f"{m}_roll{w}g": f"bp_roll{w}g_{m}"
        for w in windows
        for m in available_metrics
        if f"{m}_roll{w}g" in ballpark_rolling_calc.columns
    }
    ballpark_rolling_df = ballpark_rolling_calc.rename(columns=rename_map)

    # Add keys back for merging - ensure keys exist in original df
    key_cols = [group_col, date_col]
    if all(col in pitcher_hist_df.columns for col in key_cols):
        ballpark_rolling_df[key_cols] = pitcher_hist_df[key_cols]
    else:
        logger.error(f"Missing key columns {key_cols} in pitcher_hist_df for ballpark rolling features.")
        return pd.DataFrame(), {} # Return empty if keys are missing

    logger.info(f"Finished calculating ballpark rolling features. Found {len(ballpark_rolling_df.columns) - len(key_cols)} features.")
    return ballpark_rolling_df, rename_map


def merge_ballpark_features_historical(
    final_features_df: pd.DataFrame,
    ballpark_rolling_df: pd.DataFrame,
    bpark_rename_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Merges historical ballpark rolling features using merge_asof.

    Args:
        final_features_df: The main features DataFrame being built.
        ballpark_rolling_df: DataFrame containing ballpark rolling features.
        bpark_rename_map: The rename map used for ballpark features.

    Returns:
        The final_features_df with ballpark features merged.
    """
    if ballpark_rolling_df is None or ballpark_rolling_df.empty:
        logger.warning("Ballpark rolling features DataFrame is empty, skipping merge.")
        return final_features_df

    logger.debug("Merging historical ballpark rolling features...")

    # Use keys from bpark_rename_map to get the actual columns to merge
    bpark_roll_cols_to_merge = list(bpark_rename_map.values())
    bpark_roll_cols_to_merge = [col for col in bpark_roll_cols_to_merge if col in ballpark_rolling_df.columns]

    if not bpark_roll_cols_to_merge:
        logger.warning("No ballpark rolling columns identified to merge.")
        return final_features_df

    # Ensure required columns for merge exist in both dataframes
    if 'ballpark' not in final_features_df.columns:
        logger.error("Missing 'ballpark' column in final_features_df for ballpark merge.")
        return final_features_df
    if 'ballpark' not in ballpark_rolling_df.columns or 'game_date' not in ballpark_rolling_df.columns:
         logger.error("Missing 'ballpark' or 'game_date' column in ballpark_rolling_df for merge.")
         return final_features_df

    # Prepare for merge_asof
    final_features_df_sorted = final_features_df.sort_values('game_date')
    right_merge_cols_bp = ['ballpark', 'game_date'] + bpark_roll_cols_to_merge
    ballpark_rolling_df_sorted = ballpark_rolling_df[right_merge_cols_bp].sort_values('game_date')

    # Add merge keys safely using .copy() to avoid SettingWithCopyWarning
    final_features_df_sorted = final_features_df_sorted.copy()
    ballpark_rolling_df_sorted = ballpark_rolling_df_sorted.copy()
    final_features_df_sorted['merge_key_ballpark_left'] = final_features_df_sorted['ballpark'].astype(str)
    ballpark_rolling_df_sorted['merge_key_ballpark_right'] = ballpark_rolling_df_sorted['ballpark'].astype(str)

    try:
        merged_df = pd.merge_asof(
            final_features_df_sorted,
            ballpark_rolling_df_sorted,
            on='game_date',
            left_by='merge_key_ballpark_left',
            right_by='merge_key_ballpark_right',
            direction='backward',
            allow_exact_matches=False
        )
        # Drop merge keys
        merged_df = merged_df.drop(columns=['merge_key_ballpark_left', 'merge_key_ballpark_right'], errors='ignore')
        logger.debug("Successfully merged historical ballpark features.")
        return merged_df
    except Exception as e:
        logger.error(f"Error during historical ballpark feature merge_asof: {e}", exc_info=True)
        return final_features_df # Return original df on error


def merge_ballpark_features_prediction(
    final_features_df: pd.DataFrame,
    latest_ballpark_rolling: pd.DataFrame,
    bpark_rename_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Merges the latest ballpark rolling features onto the prediction baseline.

    Args:
        final_features_df: The prediction baseline DataFrame.
        latest_ballpark_rolling: DataFrame with the latest rolling stats per ballpark.
        bpark_rename_map: The rename map used for ballpark features.

    Returns:
        The final_features_df with latest ballpark features merged.
    """
    if latest_ballpark_rolling is None or latest_ballpark_rolling.empty:
        logger.warning("Latest ballpark rolling DataFrame is empty, skipping merge.")
        return final_features_df
    if 'ballpark' not in final_features_df.columns:
        logger.error("Missing 'ballpark' in prediction baseline for merge.")
        return final_features_df
    if 'ballpark' not in latest_ballpark_rolling.columns:
         logger.error("Missing 'ballpark' key in latest ballpark rolling data.")
         return final_features_df

    logger.debug("Merging latest ballpark features for prediction...")

    # Use keys from bpark_rename_map to get the actual columns to merge
    bp_cols_to_merge = ['ballpark'] + list(bpark_rename_map.values())
    # Check columns exist
    bp_cols_to_merge = [col for col in bp_cols_to_merge if col in latest_ballpark_rolling.columns]

    if len(bp_cols_to_merge) <= 1: # Only 'ballpark' key present
        logger.warning("No latest ballpark rolling columns identified to merge.")
        return final_features_df

    try:
        merged_df = pd.merge(
            final_features_df,
            latest_ballpark_rolling[bp_cols_to_merge],
            on='ballpark',
            how='left'
        )
        logger.debug("Successfully merged latest ballpark features.")
        return merged_df
    except Exception as e:
        logger.error(f"Error during prediction ballpark feature merge: {e}", exc_info=True)
        return final_features_df # Return original on error