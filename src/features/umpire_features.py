# src/features/umpire_features.py
import pandas as pd
import numpy as np
import logging
from typing import Callable, List, Dict, Tuple # Keep Tuple for return type consistency

# Path setup is assumed to be handled by the calling script (generate_features.py)
# or by the project structure if installed.
# DBConnection and DBConfig imports are removed as load_team_mapping_from_db is removed.

logger = logging.getLogger(__name__)

# The load_team_mapping_from_db function is removed as it's no longer needed.
# Umpire identification and team context are expected to be in the main_game_df.

def calculate_umpire_rolling_features(
    main_game_df: pd.DataFrame,
    group_col: str, # This will be 'home_plate_umpire'
    date_col: str,  # This will be 'game_date'
    metrics: List[str], # Pitcher metrics to calculate rolling averages for, per umpire
    windows: List[int],
    min_periods: int,
    calculate_multi_window_rolling: Callable
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Calculates rolling features for home plate umpires based on pitcher stats
    in games they officiated. Assumes 'home_plate_umpire' and relevant metrics
    are present in main_game_df.

    Args:
        main_game_df: DataFrame containing game-level data, including
                      the umpire identifier (group_col, e.g., 'home_plate_umpire'),
                      the date column (date_col, e.g., 'game_date'),
                      and the pitcher performance metrics to be rolled.
        group_col: The column name for the umpire identifier.
        date_col: The column name for the game date.
        metrics: List of metric column names (pitcher stats) to calculate rolling stats for.
        windows: List of window sizes for rolling calculations.
        min_periods: Minimum number of observations in a window to produce a value.
        calculate_multi_window_rolling: A callable function that performs the
                                         multi-window rolling calculation.
                                         Expected signature:
                                         (df, group_col, date_col, metrics, windows, min_periods)

    Returns:
        A tuple containing:
            - DataFrame with umpire rolling features, indexed like the input main_game_df.
              Columns are renamed with an 'ump_' prefix (e.g., 'ump_roll5g_k_percent').
            - Dictionary mapping original rolling column names to renamed umpire column names.
              Returns empty DataFrame and empty dict if input is invalid or processing fails.
    """
    if main_game_df is None or main_game_df.empty:
        logger.warning("Input DataFrame for umpire rolling features (main_game_df) is invalid or empty.")
        return pd.DataFrame(index=main_game_df.index if main_game_df is not None else None), {}

    # --- Check Required Columns ---
    required_cols = [group_col, date_col] + metrics
    missing_cols = [col for col in required_cols if col not in main_game_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in main_game_df for umpire features: {missing_cols}. Cannot proceed.")
        return pd.DataFrame(index=main_game_df.index), {}
    
    if main_game_df[group_col].isnull().all():
        logger.warning(f"The umpire identifier column '{group_col}' contains all NaN values. No umpire features can be calculated.")
        return pd.DataFrame(index=main_game_df.index), {}


    logger.info(f"Calculating umpire rolling features for '{group_col}' (Windows: {windows})...")

    # Ensure data types are correct for processing, especially date_col
    df_copy = main_game_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        try:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting date column '{date_col}' to datetime: {e}. Cannot proceed.")
            return pd.DataFrame(index=main_game_df.index), {}
    
    # Drop rows where critical columns for grouping/rolling are NaN
    # group_col (umpire) NaNs will lead to them being excluded from rolling stats for any specific umpire.
    # date_col NaNs would break time-series rolling.
    # metric NaNs are handled by the rolling function's min_periods.
    df_copy = df_copy.dropna(subset=[date_col, group_col])
    if df_copy.empty:
        logger.warning(f"DataFrame became empty after dropping NaNs in '{date_col}' or '{group_col}'. No umpire features calculated.")
        return pd.DataFrame(index=main_game_df.index), {}


    # Filter out metrics that are not actually in the DataFrame after all checks
    available_metrics = [m for m in metrics if m in df_copy.columns]
    if not available_metrics:
        logger.warning("No specified umpire metrics available in the DataFrame after pre-processing. No umpire features calculated.")
        return pd.DataFrame(index=main_game_df.index), {}

    # --- Calculate Rolling Features ---
    # The `calculate_multi_window_rolling` function is expected to handle the actual rolling logic.
    # It will group by `group_col` (home_plate_umpire) and sort by `date_col`.
    logger.info(f"Calculating rolling features for {len(df_copy[group_col].unique())} unique umpires on {len(df_copy)} rows using metrics: {available_metrics}.")
    
    try:
        umpire_rolling_calc_df = calculate_multi_window_rolling(
            df=df_copy, # Use the preprocessed df_copy
            group_col=group_col,
            date_col=date_col,
            metrics=available_metrics,
            windows=windows,
            min_periods=min_periods
        )
    except Exception as e:
        logger.error(f"Error during 'calculate_multi_window_rolling' for umpire features: {e}", exc_info=True)
        return pd.DataFrame(index=main_game_df.index), {}

    if umpire_rolling_calc_df.empty:
        logger.warning("Rolling feature calculation returned an empty DataFrame for umpires.")
        # Return empty DataFrame aligned with the original input
        return pd.DataFrame(index=main_game_df.index), {}

    # Rename columns with 'ump_' prefix
    rename_map = {
        f"{m}_roll{w}g": f"ump_roll{w}g_{m}"
        for w in windows
        for m in available_metrics
        if f"{m}_roll{w}g" in umpire_rolling_calc_df.columns # Check if column was actually created
    }
    umpire_rolling_renamed_df = umpire_rolling_calc_df.rename(columns=rename_map)

    # The calculate_multi_window_rolling function should return a DataFrame
    # that can be directly joined or reindexed to the original main_game_df.
    # If it preserves the index from df_copy, reindexing to main_game_df.index handles
    # rows that might have been dropped in df_copy or aligns with the full dataset.
    umpire_rolling_final_df = umpire_rolling_renamed_df.reindex(main_game_df.index)

    # Log count of features actually created based on rename_map keys present in output
    created_feature_count = len([col for col in rename_map.values() if col in umpire_rolling_final_df.columns])

    logger.info(f"Finished calculating umpire rolling features. Found {created_feature_count} features.")
    return umpire_rolling_final_df, rename_map