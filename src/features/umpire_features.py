# src/features/umpire_features.py
import pandas as pd
import numpy as np
import logging
from typing import Callable, List, Dict, Tuple

logger = logging.getLogger(__name__)

# Function reverted to original logic
def calculate_umpire_rolling_features(
    pitcher_hist_df: pd.DataFrame,
    umpire_hist_df: pd.DataFrame,
    group_col: str, # Should be 'home_plate_umpire'
    date_col: str,
    metrics: List[str], # e.g., ['k_percent'] derived from pitcher_hist_df
    windows: List[int],
    min_periods: int,
    calculate_multi_window_rolling: Callable
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Calculates rolling features for home plate umpires based on pitcher stats
    in games they officiated. (Reverted to original logic)

    Args:
        pitcher_hist_df: DataFrame with pitcher game stats including game_date,
                         home_team, away_team, and the metrics to roll.
        umpire_hist_df: DataFrame with historical umpire assignments including
                        game_date, home_plate_umpire, home_team, away_team.
        group_col: Column to group by ('home_plate_umpire').
        date_col: Date column ('game_date').
        metrics: List of metrics from pitcher_hist_df to calculate rolling stats for.
        windows: List of window sizes.
        min_periods: Minimum periods for rolling calculation.
        calculate_multi_window_rolling: The shared rolling calculation function.

    Returns:
        A tuple containing:
            - DataFrame with rolling umpire features, indexed like pitcher_hist_df.
              Columns renamed with 'ump_' prefix.
            - Dictionary mapping original rolling column names to renamed umpire column names.
              Returns empty DataFrame and dict if inputs are invalid or merge fails.
    """
    if pitcher_hist_df is None or pitcher_hist_df.empty or umpire_hist_df is None or umpire_hist_df.empty:
        logger.warning("Input DataFrames for umpire rolling features are invalid or empty.")
        return pd.DataFrame(index=pitcher_hist_df.index if pitcher_hist_df is not None else None), {}

    # Ensure necessary columns exist
    required_pitcher_cols = [date_col, 'home_team', 'away_team'] + metrics
    required_umpire_cols = [date_col, 'home_team', 'away_team', group_col] # group_col is 'home_plate_umpire'

    # Check columns in the input dataframes
    missing_pitcher_cols = [col for col in required_pitcher_cols if col not in pitcher_hist_df.columns]
    missing_umpire_cols = [col for col in required_umpire_cols if col not in umpire_hist_df.columns]

    if missing_pitcher_cols:
        logger.error(f"Missing required columns in pitcher_hist_df for umpire features: {missing_pitcher_cols}")
        return pd.DataFrame(index=pitcher_hist_df.index), {}
    if missing_umpire_cols:
        logger.error(f"Missing required columns in umpire_hist_df for umpire features: {missing_umpire_cols}")
        return pd.DataFrame(index=pitcher_hist_df.index), {}

    logger.info(f"Calculating umpire rolling features for '{group_col}' (Windows: {windows})...")

    # Prepare umpire data for merge (select relevant columns and drop duplicates)
    umpire_merge_df = umpire_hist_df[required_umpire_cols].drop_duplicates().copy()
    # Ensure date types match before merge
    try:
        umpire_merge_df[date_col] = pd.to_datetime(umpire_merge_df[date_col])
        # Assuming pitcher_hist_df[date_col] is already datetime from generate_features
        pitcher_hist_df_copy = pitcher_hist_df[[date_col, 'home_team', 'away_team'] + metrics].copy()
        if not pd.api.types.is_datetime64_any_dtype(pitcher_hist_df_copy[date_col]):
             pitcher_hist_df_copy[date_col] = pd.to_datetime(pitcher_hist_df_copy[date_col])
    except Exception as e:
        logger.error(f"Failed to convert date columns for umpire feature merge: {e}")
        return pd.DataFrame(index=pitcher_hist_df.index), {}

    # Merge umpire name onto pitcher history using temporary copy
    # Use left merge to keep all pitcher appearances
    merged_df = pd.merge(
        pitcher_hist_df_copy,
        umpire_merge_df,
        on=[date_col, 'home_team', 'away_team'],
        how='left'
    )
    # Align index with original pitcher_hist_df BEFORE calculating rolling stats
    merged_df = merged_df.set_index(pitcher_hist_df.index)

    # Check for missing umpires after merge
    missing_ump_count = merged_df[group_col].isnull().sum()
    if missing_ump_count > 0:
        logger.warning(f"Could not find home plate umpire for {missing_ump_count} pitcher appearances.")
        # Keep NaNs, rolling function should handle them

    # Calculate rolling features on the merged DataFrame, grouped by umpire
    available_metrics = [m for m in metrics if m in merged_df.columns]
    if not available_metrics:
        logger.warning("No specified umpire metrics found in the merged DataFrame.")
        return pd.DataFrame(index=pitcher_hist_df.index), {} # Return empty df aligned with original pitcher df

    # IMPORTANT: Pass the merged_df which is now aligned with pitcher_hist_df's index
    # Drop rows where umpire is missing *only for the calculation*, results are reindexed later
    valid_umpire_merged_df = merged_df.dropna(subset=[group_col])
    if valid_umpire_merged_df.empty:
        logger.warning(f"No rows with valid umpire names found after merge. Cannot calculate umpire features.")
        return pd.DataFrame(index=pitcher_hist_df.index), {}

    umpire_rolling_calc = calculate_multi_window_rolling(
        df=valid_umpire_merged_df, # Calculate on rows with umpires
        group_col=group_col,
        date_col=date_col,
        metrics=available_metrics,
        windows=windows,
        min_periods=min_periods
    )

    # Define and apply rename map
    rename_map = {
        f"{m}_roll{w}g": f"ump_roll{w}g_{m}"
        for w in windows
        for m in available_metrics
        if f"{m}_roll{w}g" in umpire_rolling_calc.columns
    }
    umpire_rolling_renamed = umpire_rolling_calc.rename(columns=rename_map)

    # Reindex the results back to the original pitcher_hist_df index
    # This aligns the calculated features back to the main DataFrame structure and fills non-matched indices with NaN
    umpire_rolling_final = umpire_rolling_renamed.reindex(pitcher_hist_df.index)

    logger.info(f"Finished calculating umpire rolling features. Found {len(umpire_rolling_final.columns)} features.")
    return umpire_rolling_final, rename_map