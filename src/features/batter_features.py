# src/features/batter_features.py (Updated)
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection
from src.config import StrikeoutModelConfig

# Setup logger
logger = setup_logger('batter_features')

def create_batter_features(df=None, dataset_type="all"):
    """Create batter features from pre-aggregated game-level data,
    including rolling plate discipline metrics.

    Args:
        df (pandas.DataFrame, optional): Game-level data to process.
            Expects output from the updated aggregate_batters.py, including
            columns like 'chase_percent', 'zone_swing_percent', 'contact_percent',
            'k_percent', 'woba', 'swinging_strike_percent', etc.
            If None, load from game_level_batters DB table.
        dataset_type (str): Type of dataset ("train", "test", or "all")

    Returns:
        pandas.DataFrame: DataFrame with batter features including rolling metrics.
    """
    try:
        # Load game-level batter data if not provided
        if df is None:
            logger.info(f"Loading game_level_batters data for {dataset_type} set...")
            table_to_load = 'game_level_batters' # Load the base table
            with DBConnection() as conn:
                query = f"SELECT * FROM {table_to_load}"
                # If splitting train/test here (alternative to engineer_features.py split)
                # train_seasons = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
                # test_seasons = StrikeoutModelConfig.DEFAULT_TEST_YEARS
                # seasons_to_use = train_seasons if dataset_type == "train" else test_seasons if dataset_type == "test" else None
                # if seasons_to_use:
                #     query += f" WHERE season IN {seasons_to_use}"

                df = pd.read_sql_query(query, conn)

            if df.empty:
                logger.error(f"No game-level batter data found in {table_to_load}. Run aggregate_batters.py first.")
                return pd.DataFrame()

            logger.info(f"Loaded {len(df)} rows of game-level batter data for {dataset_type}")

        # --- Preprocessing ---
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['batter_id', 'game_date']).reset_index(drop=True) # Ensure sorting

        # --- Feature Calculation ---
        logger.info(f"Calculating rolling features for {dataset_type} batter set (incl. counts)...")

        # Define count categories based on aggregate_batters output
        count_categories = ['ahead', 'behind', 'even', '0-0', '3-2']
        count_metrics = ['k_pct', 'woba', 'pa'] # Metrics aggregated per count

        # Add count-specific metrics to the list for rolling averages
        metrics_for_rolling = [
            'k_percent', 'woba', 'swinging_strike_percent',
            'chase_percent', 'zone_swing_percent', 'contact_percent',
            'zone_contact_percent', 'fastball_whiff_pct', 'breaking_whiff_pct',
            'offspeed_whiff_pct',
        ]
        # Dynamically add count metrics if they exist from aggregation
        for metric in count_metrics:
             for cat in count_categories:
                  col_name = f'{metric}_{cat}' # e.g., k_pct_ahead
                  if col_name in df.columns:
                       metrics_for_rolling.append(col_name)
                  # else: logger.warning(f"Count metric '{col_name}' not found in input.") # Optional warning


        # Verify columns exist
        cols_to_process = [m for m in metrics_for_rolling if m in df.columns]
        missing_cols = [m for m in metrics_for_rolling if m not in df.columns]
        if missing_cols: logger.warning(f"Missing base columns for rolling features: {missing_cols}.")
        if not cols_to_process:
             logger.error("No valid columns found for rolling features.")
             return df

        window_sizes = StrikeoutModelConfig.WINDOW_SIZES
        grouped_batter = df.groupby('batter_id')

        for metric in cols_to_process:
            # Shift and calculate rolling mean/std (same logic as before)
            shifted_metric = grouped_batter[metric].shift(1)
            for window in window_sizes:
                # Rolling Mean
                roll_mean_col = f'rolling_{window}g_{metric}'
                rolling_mean = shifted_metric.rolling(window, min_periods=max(1, window // 2)).mean()
                df[roll_mean_col] = rolling_mean.reset_index(level=0, drop=True)
                # Rolling Std Dev
                roll_std_col = f'rolling_{window}g_{metric}_std'
                rolling_std = shifted_metric.rolling(window, min_periods=max(2, window // 2)).std()
                df[roll_std_col] = rolling_std.reset_index(level=0, drop=True)

        # --- Rolling Splits (LHB/RHB) ---
        # This logic was previously in engineer_features.py, consolidating here
        split_metrics = ['k_pct', 'woba'] # Metrics that have _vs_R and _vs_L versions
        for metric in split_metrics:
            for hand in ['R', 'L']:
                split_col = f'{metric}_vs_{hand}'
                if split_col not in df.columns:
                    logger.warning(f"Split column {split_col} not found in batter data. Cannot create rolling features for it.")
                    continue

                # IMPORTANT: Shift the split column *before* calculating rolling mean
                shifted_split_col = grouped_batter[split_col].shift(1)

                for window in window_sizes:
                    rolling_col_name = f'rolling_{window}g_{split_col}'
                    # Calculate rolling mean on the SHIFTED data
                    rolling_mean = shifted_split_col.rolling(window, min_periods=max(1, window // 2)).mean()
                    df[rolling_col_name] = rolling_mean.reset_index(level=0, drop=True)

                    # Optionally add rolling std for splits too
                    rolling_std_col_name = f'rolling_{window}g_{split_col}_std'
                    rolling_std = shifted_split_col.rolling(window, min_periods=max(2, window // 2)).std()
                    df[rolling_std_col_name] = rolling_std.reset_index(level=0, drop=True)

        # --- Fill Missing Values ---
        # Fill NaNs only for the newly created rolling feature columns
        newly_created_cols = [col for col in df.columns if col.startswith('rolling_')]
        logger.info(f"Filling NaNs in {len(newly_created_cols)} rolling feature columns...")

        for col in newly_created_cols:
            if df[col].isnull().any():
                # Using median fill for robustness against outliers
                median_fill = df[col].median()
                df[col].fillna(median_fill, inplace=True)
                # logger.debug(f"Filled NaNs in {col} with median {median_fill:.4f}")

        # --- Save to Database ---
        table_name = "batter_predictive_features"
        if dataset_type == "train":
            table_name = "train_batter_predictive_features"
        elif dataset_type == "test":
            table_name = "test_batter_predictive_features"

        with DBConnection() as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(df)} enhanced batter features for {dataset_type} to {table_name}")

        return df

    except Exception as e:
        logger.error(f"Error creating batter features for {dataset_type}: {str(e)}", exc_info=True)
        return pd.DataFrame()

if __name__ == "__main__":
     # Example usage: Create features for all data in game_level_batters
     logger.info("Running batter feature creation directly for testing (using all data)...")
     test_features = create_batter_features(dataset_type="all")
     if not test_features.empty:
         logger.info("Test run completed. Showing sample features:")
         print(test_features.head())
         # Select columns related to a specific metric to check rolling values
         print("\nSample rolling chase_percent features:")
         chase_cols = [col for col in test_features.columns if 'chase_percent' in col]
         print(test_features[['batter_id', 'game_date'] + chase_cols].head(10))
         print("\nSample rolling k_pct_vs_R features:")
         kpct_r_cols = [col for col in test_features.columns if 'k_pct_vs_R' in col]
         print(test_features[['batter_id', 'game_date'] + kpct_r_cols].head(10))
         print(test_features.info())
     else:
         logger.error("Test run failed.")