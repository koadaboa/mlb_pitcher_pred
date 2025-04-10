# src/features/advanced_pitch_features.py
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
from src.config import StrikeoutModelConfig # Assuming window sizes are here

# Setup logger
logger = setup_logger('advanced_pitch_features')

def create_advanced_pitcher_features():
    """
    Calculates advanced pitcher features based on lagged game-level data
    to prevent data leakage. Loads data from game_level_pitchers.

    Returns:
        pandas.DataFrame: DataFrame with pitcher_id, game_pk, game_date,
                          and the newly calculated advanced features.
                          Returns empty DataFrame on error.
    """
    logger.info("Starting advanced pitcher feature creation...")
    try:
        with DBConnection() as conn:
            # Load the base game-level data
            logger.info("Loading game_level_pitchers data...")
            query = "SELECT * FROM game_level_pitchers"
            df = pd.read_sql_query(query, conn)
            logger.info(f"Loaded {len(df)} rows from game_level_pitchers")

        if df.empty:
            logger.error("game_level_pitchers table is empty or could not be loaded.")
            return pd.DataFrame()

        # Ensure correct data types and sorting
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['pitcher_id', 'game_date']).reset_index(drop=True)

        # --- Feature Calculation ---
        # Store new features in a dictionary first
        advanced_features = {}
        advanced_features['pitcher_id'] = df['pitcher_id']
        advanced_features['game_pk'] = df['game_pk']
        advanced_features['game_date'] = df['game_date']

        # Define metrics and windows for calculations
        metrics_to_process = {
            'k_per_9': {'windows': StrikeoutModelConfig.WINDOW_SIZES, 'ewm_span': 5},
            'k_percent': {'windows': StrikeoutModelConfig.WINDOW_SIZES, 'ewm_span': 5},
            'swinging_strike_percent': {'windows': StrikeoutModelConfig.WINDOW_SIZES, 'ewm_span': 5},
            'avg_velocity': {'windows': StrikeoutModelConfig.WINDOW_SIZES, 'ewm_span': 10},
            'fastball_percent': {'windows': [3, 5, 10], 'ewm_span': 10},
            'breaking_percent': {'windows': [3, 5, 10], 'ewm_span': 10},
            'offspeed_percent': {'windows': [3, 5, 10], 'ewm_span': 10},
        }

        for metric, config in metrics_to_process.items():
            logger.debug(f"Processing advanced features for: {metric}")
            if metric not in df.columns:
                logger.warning(f"Metric {metric} not found in game_level_pitchers. Skipping.")
                continue

            # --- Strict Lagging ---
            # Create lagged version of the metric ONCE before calculations
            lagged_metric_col = f'{metric}_lag1'
            df[lagged_metric_col] = df.groupby('pitcher_id')[metric].shift(1)

            # --- Rolling Standard Deviation (Volatility) ---
            for window in config.get('windows', []):
                col_name = f'rolling_{window}g_{metric}_std_lag1'
                advanced_features[col_name] = df.groupby('pitcher_id')[lagged_metric_col]\
                                                .rolling(window, min_periods=max(2, window // 2))\
                                                .std().reset_index(level=0, drop=True)

            # --- Exponentially Weighted Moving Average (EWMA) ---
            if 'ewm_span' in config:
                span = config['ewm_span']
                col_name = f'ewma_{span}g_{metric}_lag1'
                advanced_features[col_name] = df.groupby('pitcher_id')[lagged_metric_col]\
                                                .ewm(span=span, adjust=False, min_periods=max(1, span // 2))\
                                                .mean().reset_index(level=0, drop=True)

            # --- Pitch Mix Dynamics (Change from Previous Game) ---
            if 'percent' in metric: # Apply only to percentage metrics
                lagged_metric_col_lag2 = f'{metric}_lag2'
                df[lagged_metric_col_lag2] = df.groupby('pitcher_id')[metric].shift(2)
                col_name = f'{metric}_change_lag1'
                # Calculate change between lag1 and lag2
                advanced_features[col_name] = df[lagged_metric_col] - df[lagged_metric_col_lag2]


        # --- Performance Splits ---
        # NOTE: Calculating splits vs LHB/RHB requires batter info per game.
        # This typically needs to be done during aggregation (e.g., in aggregate_pitchers.py)
        # by joining pitch-level data with batter handedness, or by joining game_level_batters here.
        # Adding a placeholder comment for now.
        logger.warning("Advanced feature: Performance splits vs LHB/RHB not implemented - requires batter data linkage.")
        # Example (if split data existed in game_level_pitchers):
        # df['k_pct_vs_lhb_lag1'] = df.groupby('pitcher_id')['k_pct_vs_lhb'].shift(1)
        # advanced_features['rolling_5g_k_pct_vs_lhb_lag1'] = df.groupby('pitcher_id')['k_pct_vs_lhb_lag1']...

        # Convert dictionary to DataFrame
        advanced_features_df = pd.DataFrame(advanced_features)

        # --- Handle potential NaNs introduced by shifting/rolling ---
        numeric_cols = advanced_features_df.select_dtypes(include=np.number).columns.tolist()
        # Keep essential keys out of numeric cols if they were included accidentally
        keys = ['pitcher_id', 'game_pk']
        numeric_cols = [col for col in numeric_cols if col not in keys]

        for col in numeric_cols:
            if advanced_features_df[col].isnull().any():
                median_fill = advanced_features_df[col].median() # Or use 0? Median might be safer.
                advanced_features_df[col].fillna(median_fill, inplace=True)
                logger.debug(f"Filled NaNs in {col} with median value {median_fill:.4f}")

        logger.info(f"Successfully created {len(advanced_features_df.columns) - 3} advanced features.") # -3 for id/pk/date
        return advanced_features_df

    except Exception as e:
        logger.error(f"Error creating advanced pitcher features: {str(e)}", exc_info=True)
        return pd.DataFrame()

if __name__ == "__main__":
    # Example of how to run this module directly for testing
    logger.info("Running advanced pitcher feature creation directly for testing...")
    test_features = create_advanced_pitcher_features()
    if not test_features.empty:
        logger.info("Test run completed. Showing sample features:")
        print(test_features.head())
        print(test_features.info())
        print(test_features.describe())
    else:
        logger.error("Test run failed.")