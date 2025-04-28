# src/features/pitcher_features.py
import pandas as pd
import numpy as np
import logging
from typing import Callable, List

logger = logging.getLogger(__name__) # Use standard logging practice

def calculate_pitcher_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    metrics: List[str],
    windows: List[int],
    min_periods: int,
    calculate_multi_window_rolling: Callable # Pass the helper function
) -> pd.DataFrame:
    """
    Calculates rolling features specifically for pitchers.

    Args:
        df: DataFrame containing pitcher game-level data.
        group_col: The column to group by (e.g., 'pitcher_id').
        date_col: The date column for sorting and rolling.
        metrics: List of metric columns to calculate rolling stats for.
        windows: List of window sizes.
        min_periods: Minimum number of observations in window required.
        calculate_multi_window_rolling: The function to perform the rolling calculation.

    Returns:
        DataFrame with pitcher rolling features, indexed like the input df.
        Columns are renamed with a 'p_' prefix (e.g., 'p_roll5g_k_percent').
    """
    if df is None or df.empty:
        logger.warning("Input DataFrame for pitcher rolling features is empty.")
        return pd.DataFrame(index=df.index if df is not None else None)

    logger.info(f"Calculating pitcher rolling features (Windows: {windows})...")
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        logger.warning("No specified pitcher metrics found in the DataFrame.")
        return pd.DataFrame(index=df.index)

    pitcher_rolling_df = calculate_multi_window_rolling(
        df=df,
        group_col=group_col,
        date_col=date_col,
        metrics=available_metrics,
        windows=windows,
        min_periods=min_periods
    )

    # Rename columns with 'p_' prefix
    rename_map = {
        f"{m}_roll{w}g": f"p_roll{w}g_{m}"
        for w in windows
        for m in available_metrics
        if f"{m}_roll{w}g" in pitcher_rolling_df.columns
    }
    pitcher_rolling_df = pitcher_rolling_df.rename(columns=rename_map)
    logger.info(f"Finished calculating pitcher rolling features. Found {len(pitcher_rolling_df.columns)} features.")
    return pitcher_rolling_df

def calculate_pitcher_rest_days(pitcher_hist_df: pd.DataFrame) -> pd.Series:
    """
    Calculates the number of days since the pitcher's last appearance.

    Args:
        pitcher_hist_df: DataFrame containing pitcher game history,
                         must include 'pitcher_id' and 'game_date' columns.

    Returns:
        A pandas Series containing the days rest for each appearance,
        aligned with the input DataFrame's index. Returns empty Series if input is empty.
    """
    if pitcher_hist_df is None or pitcher_hist_df.empty:
        logger.warning("Input DataFrame for pitcher rest days is empty.")
        return pd.Series(dtype=float) # Return empty series of correct type

    if not all(col in pitcher_hist_df.columns for col in ['pitcher_id', 'game_date']):
        logger.error("Missing 'pitcher_id' or 'game_date' for rest days calculation.")
        return pd.Series(dtype=float)

    logger.info("Calculating pitcher days rest...")
    # Ensure game_date is datetime
    pitcher_hist_df_copy = pitcher_hist_df[['pitcher_id', 'game_date']].copy()
    pitcher_hist_df_copy['game_date'] = pd.to_datetime(pitcher_hist_df_copy['game_date'])

    # Sort before calculating diff
    pitcher_hist_df_sorted = pitcher_hist_df_copy.sort_values(by=['pitcher_id', 'game_date'])

    # Calculate diff within groups
    days_rest = pitcher_hist_df_sorted.groupby('pitcher_id')['game_date'].diff().dt.days

    # Reindex to match the original DataFrame's index before returning
    days_rest = days_rest.reindex(pitcher_hist_df.index)
    logger.info("Finished calculating pitcher days rest.")
    return days_rest