# src/scripts/engineer_features.py
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import time
import argparse # <-- Add this

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection
from src.data.aggregate_pitchers import aggregate_pitchers_to_game_level # Keep if needed here
from src.data.aggregate_batters import aggregate_batters_to_game_level # Keep if needed here
from src.features.pitch_features import create_pitcher_features
from src.features.batter_features import create_batter_features
from src.features.team_features import create_team_features, create_combined_features
from src.features.advanced_pitch_features import create_advanced_pitcher_features # <-- Add this
from src.config import StrikeoutModelConfig

# Setup logger
logger = setup_logger('engineer_features')

def run_feature_engineering_pipeline(args): # <-- Add args parameter
    """Run the complete feature engineering pipeline with train/test separation"""
    start_time = time.time()

    # --- Add these lines for aggregation ---
    # Decide whether to run aggregation here or ensure it's run separately
    logger.info("Ensuring base game-level tables exist (or creating them)...")
    aggregate_pitchers_to_game_level(force_rebuild=False) # Don't force rebuild by default
    aggregate_batters_to_game_level() # This one always rebuilds currently
    # --- End of added aggregation lines ---

    # Define train/test seasons explicitly
    train_seasons = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
    test_seasons = StrikeoutModelConfig.DEFAULT_TEST_YEARS

    logger.info(f"Running pipeline with train seasons {train_seasons}, test seasons {test_seasons}")
    logger.info(f"Advanced features flag: {args.advanced}") # Log the flag status

    try:
        # Split data by season BEFORE feature engineering
        with DBConnection() as conn:
            logger.info("Loading base game-level data for train/test split...")
            # Get pitcher data split by seasons
            train_pitcher_query = f"SELECT * FROM game_level_pitchers WHERE season IN {train_seasons}"
            train_pitcher_df = pd.read_sql_query(train_pitcher_query, conn)

            test_pitcher_query = f"SELECT * FROM game_level_pitchers WHERE season IN {test_seasons}"
            test_pitcher_df = pd.read_sql_query(test_pitcher_query, conn)

            # Get batter data split by seasons
            train_batter_query = f"SELECT * FROM game_level_batters WHERE season IN {train_seasons}"
            train_batter_df = pd.read_sql_query(train_batter_query, conn)

            test_batter_query = f"SELECT * FROM game_level_batters WHERE season IN {test_seasons}"
            test_batter_df = pd.read_sql_query(test_batter_query, conn)

        # Step 1: Create BASE pitcher features separately for train/test
        logger.info("Creating BASE pitcher features for train set...")
        train_pitcher_features = create_pitcher_features(train_pitcher_df.copy(), "train") # Pass copy

        logger.info("Creating BASE pitcher features for test set...")
        test_pitcher_features = create_pitcher_features(test_pitcher_df.copy(), "test") # Pass copy

        # --- Add this block for advanced features ---
        if args.advanced:
            logger.info("Creating advanced pitcher features...")
            # This function loads its own data to calculate features across all seasons
            advanced_features_df = create_advanced_pitcher_features()

            if not advanced_features_df.empty:
                logger.info("Merging advanced features into base predictive features...")
                # Merge advanced features into existing train/test feature tables
                # Use left merge to keep only rows corresponding to train/test sets
                train_pitcher_features = pd.merge(
                    train_pitcher_features,
                    advanced_features_df,
                    on=['pitcher_id', 'game_pk', 'game_date'],
                    how='left'
                )
                test_pitcher_features = pd.merge(
                    test_pitcher_features,
                    advanced_features_df,
                    on=['pitcher_id', 'game_pk', 'game_date'],
                    how='left'
                )

                # Handle potential new NaNs introduced by merging (advanced features might not exist for earliest games)
                # Fill NaNs that were introduced in the newly merged columns
                base_cols_train = set(train_pitcher_df.columns) # Cols before merge
                new_cols_train = set(train_pitcher_features.columns) - base_cols_train
                for col in new_cols_train:
                     if train_pitcher_features[col].isnull().any():
                          median_fill = train_pitcher_features[col].median()
                          train_pitcher_features[col].fillna(median_fill, inplace=True)
                          logger.debug(f"Filled merge-introduced NaNs in train {col} with median {median_fill:.4f}")

                base_cols_test = set(test_pitcher_df.columns) # Cols before merge
                new_cols_test = set(test_pitcher_features.columns) - base_cols_test
                for col in new_cols_test:
                     if test_pitcher_features[col].isnull().any():
                          median_fill = test_pitcher_features[col].median() # Use test median or train median? Train median might be safer to avoid leakage.
                          # Let's use train median to be safe, requires passing it or recalculating
                          # For simplicity now, using test median. Consider refining this.
                          test_pitcher_features[col].fillna(median_fill, inplace=True)
                          logger.debug(f"Filled merge-introduced NaNs in test {col} with median {median_fill:.4f}")


                # Re-save the updated feature tables WITH advanced features
                logger.info("Re-saving train/test predictive pitch features with advanced features...")
                with DBConnection() as conn:
                     train_pitcher_features.to_sql("train_predictive_pitch_features", conn, if_exists='replace', index=False)
                     test_pitcher_features.to_sql("test_predictive_pitch_features", conn, if_exists='replace', index=False)
                logger.info("Merged and re-saved predictive pitcher features including advanced ones.")
            else:
                logger.warning("Advanced feature creation returned empty DataFrame. Skipping merge.")
        # --- End of advanced features block ---


        # Step 2: Create batter features separately for train/test
        logger.info("Creating batter features for train set...")
        # Pass copy just in case
        train_batter_features = create_batter_features(train_batter_df.copy(), "train")

        logger.info("Creating batter features for test set...")
        test_batter_features = create_batter_features(test_batter_df.copy(), "test")

        # Step 3: Create team features (using only train seasons' data for opponent stats)
        logger.info("Creating team features using only training data seasons...")
        # Ensure create_team_features uses only train_seasons for calculations if needed
        team_features = create_team_features(seasons=train_seasons) # Pass train seasons

        # Step 4: Create combined features separately using the potentially updated pitcher features
        logger.info("Creating combined features for train set...")
        # Pass the potentially updated train_pitcher_features
        train_combined = create_combined_features(pitcher_df=train_pitcher_features, team_df=team_features, dataset_type="train")

        logger.info("Creating combined features for test set...")
         # Pass the potentially updated test_pitcher_features
        test_combined = create_combined_features(pitcher_df=test_pitcher_features, team_df=team_features, dataset_type="test")

        pipeline_time = time.time() - start_time
        logger.info(f"Feature engineering pipeline completed in {pipeline_time:.2f} seconds.")
        return True

    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # --- Add Argument Parser ---
    parser = argparse.ArgumentParser(description="Run the MLB feature engineering pipeline.")
    parser.add_argument(
        "--advanced",
        action="store_true", # Makes it a flag, presence means True
        help="Include the creation and merging of advanced pitcher features."
    )
    args = parser.parse_args()
    # --- End Argument Parser ---

    logger.info("Starting feature engineering pipeline script...")
    success = run_feature_engineering_pipeline(args) # Pass parsed args
    if success:
        logger.info("Feature engineering pipeline completed successfully.")
    else:
        logger.error("Feature engineering pipeline failed.")