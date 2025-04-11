# src/scripts/engineer_features.py
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import time
import argparse

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection
from src.data.aggregate_pitchers import aggregate_pitchers_to_game_level
from src.data.aggregate_batters import aggregate_batters_to_game_level
from src.data.aggregate_teams import aggregate_teams_to_game_level
from src.features.pitch_features import create_pitcher_features
from src.features.batter_features import create_batter_features 
from src.features.team_features import create_team_features 
from src.features.advanced_pitch_features import create_advanced_pitcher_features
from src.features.advanced_opp_features import create_advanced_opponent_features 
from src.config import StrikeoutModelConfig

logger = setup_logger('engineer_features')

def run_feature_engineering_pipeline(args):
    """Run the complete feature engineering pipeline with new aggregations"""
    start_time = time.time()

    # --- Run Aggregation Steps ---
    logger.info("Running/Verifying base game-level aggregations...")
    try:
        # Use the flag if added and working
        force_rebuild_agg = getattr(args, 'force_rebuild_agg', False)
        aggregate_pitchers_to_game_level(force_rebuild=force_rebuild_agg)
        logger.info("Pitcher aggregation check complete.")
        # Batter aggregation includes counts now, rebuild to ensure they are present
        aggregate_batters_to_game_level()
        logger.info("Batter aggregation (with splits/counts) complete.")
        # <<< Run NEW team aggregation >>>
        aggregate_teams_to_game_level()
        logger.info("Team game-level aggregation complete.")

    except Exception as e:
        logger.error(f"Error during aggregation steps: {e}.", exc_info=True)
        return False

    try:
        # --- Load Base Data (Splitting by Season) ---
        # (Loading logic remains similar, ensure tables exist)
        with DBConnection() as conn:
            logger.info("Loading BASE game-level data for train/test split...")
            # Load pitcher data
            train_seasons = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
            # Ensure train_seasons is a tuple
            if isinstance(train_seasons, (int, str)): train_seasons = (train_seasons,)
            train_pitcher_query = f"SELECT * FROM game_level_pitchers WHERE season IN {tuple(train_seasons)}"

            test_seasons = StrikeoutModelConfig.DEFAULT_TEST_YEARS
            # Ensure test_seasons is a tuple
            if isinstance(test_seasons, (int, str)): test_seasons = (test_seasons,)
            test_pitcher_query = f"SELECT * FROM game_level_pitchers WHERE season IN {tuple(test_seasons)}"
            
            train_pitcher_df = pd.read_sql_query(train_pitcher_query, conn)
            test_pitcher_df = pd.read_sql_query(test_pitcher_query, conn)
            logger.info(f"Loaded {len(train_pitcher_df)} train pitcher games, {len(test_pitcher_df)} test pitcher games.")

            # Apply the same tuple conversion and formatting for the batter queries:
            train_batter_query = f"SELECT * FROM game_level_batters WHERE season IN {tuple(train_seasons)}"
            test_batter_query = f"SELECT * FROM game_level_batters WHERE season IN {tuple(test_seasons)}"

            # Load batter data (now includes count metrics)
            train_batter_df = pd.read_sql_query(train_batter_query, conn)
            test_batter_df = pd.read_sql_query(test_batter_query, conn)
            logger.info(f"Loaded {len(train_batter_df)} train batter games, {len(test_batter_df)} test batter games (with splits/counts).")

            # Load NEW game-level team data (used by advanced_opp_features)
            # We load all of it, feature creation function handles sorting/shifting
            adv_opp_query = "SELECT * FROM game_level_team_stats" # No, advanced_opp_features loads this itself
            # adv_opp_base_df = pd.read_sql_query(adv_opp_query, conn)

        if train_pitcher_df.empty or test_pitcher_df.empty or train_batter_df.empty or test_batter_df.empty:
             logger.error("One or more base dataframes empty after loading. Aborting.")
             return False

        # --- Step 1 & 2: Pitcher Features (Base + Optional Advanced) ---
        logger.info("Creating BASE pitcher features...")
        train_pitcher_features = create_pitcher_features(train_pitcher_df.copy(), "train")
        test_pitcher_features = create_pitcher_features(test_pitcher_df.copy(), "test")
        if args.advanced_pitcher:
            logger.info("Creating & Merging advanced pitcher features...")
            adv_pitch_features_df = create_advanced_pitcher_features()
            if not adv_pitch_features_df.empty:
                 train_pitcher_features = pd.merge(train_pitcher_features, adv_pitch_features_df, on=['pitcher_id', 'game_pk', 'game_date'], how='left')
                 test_pitcher_features = pd.merge(test_pitcher_features, adv_pitch_features_df, on=['pitcher_id', 'game_pk', 'game_date'], how='left')
                 # Add NaN filling for merged columns if needed
            else: logger.warning("Advanced pitcher features empty, skipping merge.")


        # --- Step 3: Batter Features (Now includes rolling counts) ---
        logger.info("Creating batter features (including rolling LHB/RHB splits & counts)...")
        # Pass the loaded train/test batter DFs which have the raw count metrics
        train_batter_features = create_batter_features(train_batter_df.copy(), "train")
        test_batter_features = create_batter_features(test_batter_df.copy(), "test")


        # --- Step 4: Create Team Features (Base season stats - Optional?) ---
        # Keep this if you want lagged season stats as additional features
        logger.info("Creating base team season features (using only training data seasons)...")
        team_features_base = create_team_features(seasons=train_seasons)


        # --- Step 5: Create ADVANCED Opponent Features (Game-Level, Optional) ---
        adv_opp_features_df = pd.DataFrame() # Ensure defined
        if args.advanced_opponent:
             logger.info("Creating advanced opponent features (using game-level stats)...")
             # This function creates features for all teams/games and saves to its own table
             adv_opp_features_df = create_advanced_opponent_features(window_games=10) # Pass window size
             if adv_opp_features_df.empty:
                  logger.warning("Advanced opponent feature creation failed or returned empty.")
        # NOTE: The advanced_opp_features_df now contains game_pk, team, opponent, features


        # --- Step 6: Create COMBINED Features (Train/Test) ---
        logger.info("Creating combined features for train set...")
        train_combined = create_combined_features_final(
            pitcher_features_df=train_pitcher_features,
            # No longer pass batter_features separately, opponent stats are game-level now
            base_team_features_df=team_features_base, # Pass base season features (optional)
            adv_opp_features_df=adv_opp_features_df, # Pass game-level advanced opponent features
            dataset_type="train"
        )

        logger.info("Creating combined features for test set...")
        test_combined = create_combined_features_final(
            pitcher_features_df=test_pitcher_features,
            base_team_features_df=team_features_base, # Use same train-derived base team features
            adv_opp_features_df=adv_opp_features_df,
            dataset_type="test"
        )

        pipeline_time = time.time() - start_time
        logger.info(f"Feature engineering pipeline completed in {pipeline_time:.2f} seconds.")
        return True

    except Exception as e:
        logger.error(f"Error in feature engineering pipeline: {str(e)}", exc_info=True)
        return False


# Helper function to create final combined features (can move to separate module)
def create_combined_features_final(pitcher_features_df, base_team_features_df, adv_opp_features_df, dataset_type="all"):
    """
    Combines pitcher features with relevant opponent features (base season and/or advanced game-level).
    """
    logger.info(f"Creating FINAL combined features for {dataset_type} dataset...")

    try:
        if pitcher_features_df.empty:
             logger.error("Pitcher features empty. Cannot combine.")
             return pd.DataFrame()

        combined_df = pitcher_features_df.copy()
        combined_df['game_date'] = pd.to_datetime(combined_df['game_date'])

        # Determine opponent team
        combined_df['opponent_team_name'] = combined_df.apply(
             lambda row: row['away_team'] if row['is_home'] == 1 else row['home_team'], axis=1
        )

        # 1. Merge Base Season Opponent Features (Optional)
        if not base_team_features_df.empty:
             logger.info("Merging base (season-level) opponent features...")
             combined_df = pd.merge(
                 combined_df,
                 base_team_features_df.add_prefix('opp_base_'),
                 left_on=['opponent_team_name', 'season'],
                 right_on=['opp_base_team', 'opp_base_season'],
                 how='left'
             )
             # Drop merge keys
             combined_df.drop(columns=[col for col in combined_df.columns if col.startswith('opp_base_team') or col.startswith('opp_base_season')], errors='ignore', inplace=True)


        # 2. Merge Advanced Game-Level Opponent Features
        if not adv_opp_features_df.empty:
            logger.info("Merging advanced (game-level) opponent features...")
            # Need opponent's features *before* the current game.
            # The adv_opp_features_df already contains rolling features calculated on SHIFTED data.
            # We merge based on the opponent team and the game identifier.
            combined_df = pd.merge(
                combined_df,
                adv_opp_features_df.add_prefix('opp_adv_'),
                # Merge on game_pk and the team that is the *opponent* in the pitcher_features_df perspective
                left_on=['game_pk', 'opponent_team_name'],
                right_on=['opp_adv_game_pk', 'opp_adv_team'], # opponent_team_name matches the 'team' in adv_opp_features
                how='left'
            )
            logger.info(f"Shape after merging advanced opponent features: {combined_df.shape}")
            # Drop merge keys from opponent features
            combined_df.drop(columns=[col for col in combined_df.columns if col.startswith('opp_adv_game_pk') or col.startswith('opp_adv_team') or col.startswith('opp_adv_opponent') or col.startswith('opp_adv_game_date')], errors='ignore', inplace=True)


        # 3. Handle NaNs from merges
        opp_cols = [col for col in combined_df.columns if 'opp_' in col]
        logger.info(f"Filling NaNs in {len(opp_cols)} opponent feature columns...")
        for col in opp_cols:
             if combined_df[col].isnull().any():
                  # Use median or a reasonable default (e.g., 0 for rates?)
                  median_fill = combined_df[col].median()
                  combined_df[col].fillna(median_fill, inplace=True)
                  # logger.debug(f"Filled NaNs in {col} with median {median_fill:.4f}")

        # Drop intermediate opponent name column
        combined_df.drop(columns=['opponent_team_name'], errors='ignore', inplace=True)

        # --- Save Combined Features ---
        table_name = f"{dataset_type}_combined_features"
        with DBConnection() as conn:
            combined_df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(combined_df)} final combined features for {dataset_type} to {table_name}")

        logger.info(f"Final combined features created for {dataset_type}. Final shape: {combined_df.shape}")
        return combined_df

    except Exception as e:
        logger.error(f"Error creating final combined features for {dataset_type}: {str(e)}", exc_info=True)
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MLB feature engineering pipeline.")
    parser.add_argument("--advanced-pitcher", action="store_true", help="Include advanced PITCHER features.")
    parser.add_argument("--advanced-opponent", action="store_true", help="Include advanced OPPONENT features (game-level).")
    # Add the force rebuild flag correctly
    parser.add_argument("--force-rebuild-agg", action="store_true", help="Force rebuild of pitcher aggregation table.")
    args = parser.parse_args()

    logger.info("Starting feature engineering pipeline script...")
    success = run_feature_engineering_pipeline(args)
    # ... (logging success/failure) ...