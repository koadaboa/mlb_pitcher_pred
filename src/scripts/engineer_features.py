# src/scripts/engineer_features.py

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import sys
import time
import pickle
import gc
from category_encoders import TargetEncoder # Using TargetEncoder now
from datetime import datetime, timedelta # Ensure timedelta is imported
import joblib # For loading encoder object

# Assuming script is run via python -m src.scripts.engineer_features
try:
    from src.config import DBConfig, StrikeoutModelConfig, LogConfig, FileConfig
    from src.data.utils import setup_logger, DBConnection, find_latest_file # Added find_latest_file
    # Import aggregation functions
    from src.data.aggregate_statcast import aggregate_statcast_pitchers_sql, aggregate_statcast_batters_sql
    # Import feature creation functions
    from src.features.pitcher_features import create_pitcher_features # Assuming main function is here
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    MODULE_IMPORTS_OK = False
    sys.exit(1) # Exit if essential modules are missing

# Setup logger
LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True) # Ensure log dir exists
logger = setup_logger('engineer_features', LogConfig.LOG_DIR / 'engineer_features.log', level=logging.DEBUG)

# --- Argument Parser ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Engineer Features for MLB Strikeout Prediction Model.")
    parser.add_argument("--prediction-date", type=str, default=None,
                        help="If specified, generate features only for this date (YYYY-MM-DD) for prediction.")
    parser.add_argument("--real-world", action="store_true",
                        help="Use when running for actual predictions (affects data loading/saving).")
    return parser.parse_args()

# --- Target Encoding Function ---
def apply_target_encoding(train_df, test_df, target_col, cols_to_encode, min_samples_leaf=20, smoothing=10.0):
    """Applies target encoding smoothing and saves the fitted encoder."""
    logger.info(f"Applying target encoding...")

    train_df = train_df.copy()
    test_df = test_df.copy()

    # Ensure target is numeric and handle potential NaNs before fitting encoder
    train_df[target_col] = pd.to_numeric(train_df[target_col], errors='coerce')
    test_df[target_col] = pd.to_numeric(test_df[target_col], errors='coerce')
    train_df_clean = train_df.dropna(subset=[target_col]).copy() # Fit encoder only on non-NaN target rows
    if len(train_df_clean) < len(train_df):
         logger.warning(f"Dropped {len(train_df) - len(train_df_clean)} rows with NaN target before fitting encoder.")

    logger.info(f"Global mean {target_col} (used for fit): {train_df_clean[target_col].mean():.4f}")

    # Ensure columns to encode exist and handle potential NaNs before fitting
    valid_cols_to_encode = []
    placeholder = 'MISSING_CATEGORY'
    for col in cols_to_encode:
        if col in train_df.columns:
             # Fill NaNs in categorical column with a placeholder before encoding
             train_df[col] = train_df[col].fillna(placeholder)
             test_df[col] = test_df[col].fillna(placeholder)
             train_df_clean[col] = train_df_clean[col].fillna(placeholder) # Apply to clean df too
             valid_cols_to_encode.append(col)
        else:
             logger.warning(f"Column '{col}' not found for target encoding.")

    if not valid_cols_to_encode:
         logger.warning("No valid columns found for target encoding.")
         return train_df, test_df, None # Return original dfs if no encoding happened

    encoder = TargetEncoder(cols=valid_cols_to_encode, min_samples_leaf=min_samples_leaf, smoothing=smoothing)

    # Fit on training data ONLY (with NaN target rows removed)
    encoder.fit(train_df_clean[valid_cols_to_encode], train_df_clean[target_col])

    # Transform both original train and test data (including rows that had NaN target)
    train_encoded = encoder.transform(train_df[valid_cols_to_encode])
    test_encoded = encoder.transform(test_df[valid_cols_to_encode])

    # Rename encoded columns
    encoded_cols_map = {col: f"{col}_encoded" for col in valid_cols_to_encode}
    train_encoded = train_encoded.rename(columns=encoded_cols_map) # No inplace
    test_encoded = test_encoded.rename(columns=encoded_cols_map) # No inplace

    # Combine back with original dataframes (dropping original categorical columns)
    train_df_out = pd.concat([train_df.drop(columns=valid_cols_to_encode), train_encoded], axis=1)
    test_df_out = pd.concat([test_df.drop(columns=valid_cols_to_encode), test_encoded], axis=1)

    # Save the fitted encoder object
    try:
         encoder_path = Path(FileConfig.MODELS_DIR) / f"target_encoder_{datetime.now().strftime('%Y%m%d')}.pkl"
         encoder_path.parent.mkdir(parents=True, exist_ok=True)
         with open(encoder_path, 'wb') as f:
              pickle.dump(encoder, f) # Save the entire encoder object
         logger.info(f"Saved fitted TargetEncoder object to {encoder_path}")
    except Exception as e:
         logger.error(f"Failed to save target encoder object: {e}")

    return train_df_out, test_df_out, encoder # Return encoder for optional immediate use

# --- Umpire Prediction Function ---
def predict_home_plate_umpire(game_date_dt, home_team_abbr, away_team_abbr, schedule_df, historical_umpire_df):
    """
    Predicts the home plate umpire based on series rotation.

    Args:
        game_date_dt (datetime.date): The date of the game to predict.
        home_team_abbr (str): Home team abbreviation.
        away_team_abbr (str): Away team abbreviation.
        schedule_df (pd.DataFrame): DataFrame containing master_schedule data.
        historical_umpire_df (pd.DataFrame): DataFrame with historical umpire assignments.

    Returns:
        str: Predicted home plate umpire name, or 'Unknown' if prediction fails.
    """
    try:
        if schedule_df.empty or historical_umpire_df.empty:
            logger.warning("Schedule or historical umpire data missing for prediction.")
            return "Unknown"

        game_date_str = game_date_dt.strftime('%Y-%m-%d')
        yesterday_dt = game_date_dt - timedelta(days=1)
        yesterday_str = yesterday_dt.strftime('%Y-%m-%d')

        # Find games in the series involving these teams around the target date
        # Look back a few days to establish the series pattern
        lookback_days = 4
        start_series_check_dt = game_date_dt - timedelta(days=lookback_days)

        series_games = schedule_df[
            (schedule_df['game_date'] >= start_series_check_dt) &
            (schedule_df['game_date'] <= game_date_dt) &
            (schedule_df['home_team'] == home_team_abbr) &
            (schedule_df['away_team'] == away_team_abbr)
        ].sort_values('game_date')

        if len(series_games) <= 1:
            logger.debug(f"Cannot predict umpire for {away_team_abbr} @ {home_team_abbr} on {game_date_str}: First game of series or insufficient history.")
            return "Unknown" # Likely first game of series

        # Find the game played *yesterday* within this identified series
        previous_game = series_games[series_games['game_date'] == yesterday_dt]

        if previous_game.empty:
            logger.warning(f"Cannot predict umpire for {game_date_str}: No game found yesterday ({yesterday_str}) in the identified series for {away_team_abbr} @ {home_team_abbr}.")
            # Could potentially look back 2 days if yesterday was an off-day in the series
            return "Unknown"

        # Get umpire crew from yesterday's game
        prev_game_umpires = historical_umpire_df[
            (historical_umpire_df['game_date'] == yesterday_str) &
            (historical_umpire_df['home_team'] == home_team_abbr) &
            (historical_umpire_df['away_team'] == away_team_abbr)
        ]

        if prev_game_umpires.empty:
            logger.warning(f"Cannot predict umpire for {game_date_str}: Missing umpire data for previous game ({yesterday_str}) for {away_team_abbr} @ {home_team_abbr}.")
            return "Unknown"

        # Apply rotation logic: Yesterday's 1B -> Today's HP
        # Assumes columns 'first_base_umpire', etc. exist in historical_umpire_df
        prev_1b_ump = prev_game_umpires['first_base_umpire'].iloc[0]

        if pd.isna(prev_1b_ump) or not prev_1b_ump:
            logger.warning(f"Cannot predict umpire for {game_date_str}: Previous 1B umpire name is missing for {yesterday_str}.")
            return "Unknown"

        logger.info(f"Predicted HP Umpire for {game_date_str} ({away_team_abbr} @ {home_team_abbr}): {prev_1b_ump} (was 1B yesterday)")
        return prev_1b_ump

    except Exception as e:
        logger.error(f"Error predicting umpire for {game_date_str} ({away_team_abbr} @ {home_team_abbr}): {e}", exc_info=True)
        return "Unknown"


# --- Main Feature Engineering Pipeline ---
def run_feature_pipeline(args):
    """Runs the full feature engineering pipeline."""
    start_pipeline_time = time.time()
    db_path = Path(DBConfig.PATH)
    prediction_mode = args.prediction_date is not None

    if prediction_mode:
         logger.info(f"=== Starting PREDICTION Feature Engineering for Date: {args.prediction_date} ===")
         output_table = "prediction_features" # Target table for base features
         prediction_date_dt = datetime.strptime(args.prediction_date, '%Y-%m-%d').date()

         logger.info("STEP 1 & 2: Loading supplementary data for prediction...")
         try:
              with DBConnection(db_path) as conn:
                   # Load MASTER SCHEDULE for series identification
                   schedule_query = "SELECT game_date, home_team, away_team FROM master_schedule"
                   schedule_df = pd.read_sql_query(schedule_query, conn)
                   schedule_df['game_date'] = pd.to_datetime(schedule_df['game_date']).dt.date
                   logger.info(f"Loaded {len(schedule_df)} schedule records.")

                   # Load HISTORICAL UMPIRE data (including all positions if available)
                   umpire_cols = ['game_date', 'home_team', 'away_team', 'home_plate_umpire',
                                  'first_base_umpire', 'second_base_umpire', 'third_base_umpire']
                   umpire_query = f"SELECT {', '.join(umpire_cols)} FROM espn_umpire_data WHERE DATE(game_date) < '{args.prediction_date}'"
                   historical_umpire_df = pd.read_sql_query(umpire_query, conn)
                   historical_umpire_df['game_date'] = pd.to_datetime(historical_umpire_df['game_date']).dt.strftime('%Y-%m-%d') # Standardize format
                   logger.info(f"Loaded {len(historical_umpire_df)} historical umpire records.")

                   # Load game_level_team_stats for opponent features (up to yesterday)
                   max_hist_date = (prediction_date_dt - timedelta(days=1)).strftime('%Y-%m-%d')
                   team_stats_cols = ['game_pk', 'team', 'opponent', 'game_date', 'k_percent', 'bb_percent', 'swing_percent', 'contact_percent', 'swinging_strike_percent', 'chase_percent', 'zone_contact_percent']
                   team_stats_query = f"SELECT {', '.join(team_stats_cols)} FROM game_level_team_stats WHERE DATE(game_date) <= '{max_hist_date}'"
                   game_level_team_stats = pd.read_sql_query(team_stats_query, conn)
                   logger.info(f"Loaded {len(game_level_team_stats)} historical team stat records.")

                   # Load minimal pitch data if needed for platoon (up to yesterday)
                   minimal_pitch_df = pd.DataFrame()
                   if True: # Assuming platoon features are needed
                        logger.info("Loading minimal pitch data for platoon features (prediction)...")
                        pitch_cols = ['pitcher', 'game_pk', 'game_date', 'stand', 'events']
                        pitch_query = f"SELECT {', '.join(pitch_cols)} FROM statcast_pitchers WHERE DATE(game_date) <= '{max_hist_date}'"
                        minimal_pitch_df = pd.read_sql_query(pitch_query, conn)
                        logger.info(f"Loaded {len(minimal_pitch_df)} minimal pitch records.")

                   # Load historical game_level_pitcher data for lags/trends (up to yesterday)
                   pitcher_stats_query = f"SELECT * FROM game_level_pitchers WHERE DATE(game_date) <= '{max_hist_date}'"
                   game_level_pitcher_stats = pd.read_sql_query(pitcher_stats_query, conn)
                   logger.info(f"Loaded {len(game_level_pitcher_stats)} historical pitcher stat records.")

                   # Load BASELINE pitcher data FOR THE PREDICTION DATE
                   pred_baseline_query = f"SELECT * FROM game_level_pitchers WHERE DATE(game_date) = '{args.prediction_date}'"
                   pred_baseline_df = pd.read_sql_query(pred_baseline_query, conn)
                   logger.info(f"Loaded {len(pred_baseline_df)} baseline pitcher records for {args.prediction_date}.")
                   if pred_baseline_df.empty:
                        logger.error(f"No baseline pitcher data found for prediction date {args.prediction_date}. Run data fetching/aggregation.")
                        sys.exit(1)

         except Exception as e:
              logger.error(f"Error loading supplementary data for prediction: {e}", exc_info=True)
              sys.exit(1)

         logger.info("STEP 3a: Predicting umpires for prediction date...")
         predicted_umpires = []
         # Ensure 'game_date' is datetime.date for comparison
         pred_baseline_df['game_date_dt'] = pd.to_datetime(pred_baseline_df['game_date']).dt.date
         for _, row in pred_baseline_df.iterrows():
             pred_ump = predict_home_plate_umpire(
                 row['game_date_dt'],
                 row['home_team'],
                 row['away_team'],
                 schedule_df,
                 historical_umpire_df # Pass df with all umpire positions
             )
             predicted_umpires.append({
                 'game_date': row['game_date'], # Keep original string/datetime format
                 'home_team': row['home_team'],
                 'away_team': row['away_team'],
                 'umpire': pred_ump # Predicted HP umpire
             })
         pred_baseline_df = pred_baseline_df.drop(columns=['game_date_dt']) # Drop helper column
         umpire_df_pred = pd.DataFrame(predicted_umpires)
         # Convert game_date to string to match historical umpire_df before passing
         umpire_df_pred['game_date'] = pd.to_datetime(umpire_df_pred['game_date']).dt.strftime('%Y-%m-%d')
         logger.info(f"Predicted umpires for {len(umpire_df_pred)} games.")

         logger.info("STEP 3b: Creating pitcher features for prediction date...")
         # Combine historical and prediction baseline for feature calculation context
         combined_pitcher_stats = pd.concat([game_level_pitcher_stats, pred_baseline_df], ignore_index=True)

         # Call feature creation function - pass the *predicted* umpire data
         final_pred_features = create_pitcher_features(
              pitcher_data=combined_pitcher_stats, # Pass combined data for context
              team_stats_data=game_level_team_stats,
              umpire_data=umpire_df_pred, # Pass DataFrame with predicted 'umpire' column
              pitch_data=minimal_pitch_df
         )

         # Filter results back down to only the prediction date
         final_pred_features = final_pred_features[
              pd.to_datetime(final_pred_features['game_date']).dt.strftime('%Y-%m-%d') == args.prediction_date
         ].copy()
         logger.info(f"Generated {len(final_pred_features)} rows of features for prediction date.")

         if final_pred_features.empty:
              logger.error("Feature generation resulted in empty dataframe for prediction date.")
              sys.exit(1)

         # Apply target encoding using saved encoder object
         logger.info("STEP 4: Applying target encoding using saved encoder object...")
         try:
              model_dir = Path(FileConfig.MODELS_DIR)
              encoder_path = find_latest_file(model_dir, "target_encoder_*.pkl") # Find the saved encoder object
              if not encoder_path:
                   logger.error("Could not find target encoder .pkl file in models directory. Cannot apply encoding for prediction.")
                   sys.exit(1)
              logger.info(f"Loading TargetEncoder object from: {encoder_path}")
              with open(encoder_path, 'rb') as f:
                   encoder = pickle.load(f) # Load the fitted encoder

              cols_to_encode = StrikeoutModelConfig.TARGET_ENCODING_COLS
              valid_cols_to_encode = [col for col in cols_to_encode if col in final_pred_features.columns]

              if valid_cols_to_encode:
                   logger.info(f" Applying loaded target encoding to: {valid_cols_to_encode}")
                   # Fill NaNs with placeholder before transforming
                   placeholder = 'MISSING_CATEGORY' # Must match placeholder used during training fit
                   for col in valid_cols_to_encode:
                       final_pred_features[col] = final_pred_features[col].fillna(placeholder)

                   # Use the loaded encoder's transform method
                   pred_encoded = encoder.transform(final_pred_features[valid_cols_to_encode])

                   encoded_cols_map = {col: f"{col}_encoded" for col in valid_cols_to_encode}
                   pred_encoded = pred_encoded.rename(columns=encoded_cols_map) # No inplace

                   # Combine back, dropping original columns
                   final_pred_features = pd.concat([final_pred_features.drop(columns=valid_cols_to_encode), pred_encoded], axis=1)
              else:
                   logger.warning("No valid columns found for target encoding during prediction.")

         except Exception as e:
              logger.error(f"Error applying target encoding during prediction: {e}", exc_info=True)
              sys.exit(1)

         logger.info("STEP 5: Saving prediction features...")
         try:
              with DBConnection(db_path) as conn:
                   final_pred_features.to_sql(output_table, conn, if_exists='replace', index=False)
              logger.info(f"Saved {len(final_pred_features)} prediction records to '{output_table}'")
         except Exception as e:
              logger.error(f"Error saving prediction features: {e}", exc_info=True)
              sys.exit(1)

    else: # --- Historical/Training Mode ---
         logger.info("=== Starting Historical Feature Engineering Pipeline (SQL Aggregation) ===")
         logger.info("Running aggregation and feature engineering steps (Checkpoints disabled)...")

         logger.info("STEP 1: Aggregating raw Statcast data using SQL functions...")
         try:
              game_level_pitcher_stats = aggregate_statcast_pitchers_sql()
              logger.info(f"Successfully aggregated/loaded {len(game_level_pitcher_stats)} pitcher game records.")
              if game_level_pitcher_stats.empty: raise ValueError("Pitcher aggregation returned empty DataFrame.")

              game_level_team_stats = aggregate_statcast_batters_sql()
              logger.info(f"Successfully aggregated/loaded {len(game_level_team_stats)} team game records.")
              if game_level_team_stats.empty: raise ValueError("Team aggregation returned empty DataFrame.")
         except Exception as e:
              logger.error(f"Error during data aggregation step: {e}", exc_info=True); sys.exit(1)

         logger.info("STEP 2: Loading supplementary data (Umpires, Minimal Pitch)...")
         try:
              with DBConnection(db_path) as conn:
                   # Load HISTORICAL UMPIRE data - selecting only HP Umpire and renaming
                   umpire_query = "SELECT game_date, home_team, away_team, home_plate_umpire FROM espn_umpire_data"
                   umpire_df_hist = pd.read_sql_query(umpire_query, conn)
                   # Rename for the create_pitcher_features function
                   umpire_df_hist = umpire_df_hist.rename(columns={'home_plate_umpire': 'umpire'}) # No inplace
                   logger.info(f"Loaded and prepared {len(umpire_df_hist)} historical umpire records.")

                   # Load minimal pitch data if needed
                   minimal_pitch_df = pd.DataFrame()
                   if True: # Assuming platoon features needed
                        logger.info("Loading minimal pitch data for platoon features...")
                        pitch_cols = ['pitcher', 'game_pk', 'game_date', 'stand', 'events']
                        pitch_query = f"SELECT {', '.join(pitch_cols)} FROM statcast_pitchers"
                        minimal_pitch_df = pd.read_sql_query(pitch_query, conn)
                        logger.info(f"Loaded {len(minimal_pitch_df)} minimal pitch records.")
         except Exception as e:
              logger.error(f"Error loading supplementary data: {e}", exc_info=True); sys.exit(1)

         logger.info("STEP 3: Creating pitcher features...")
         try:
              # Pass the actual historical umpire data
              combined_features = create_pitcher_features(
                   pitcher_data=game_level_pitcher_stats,
                   team_stats_data=game_level_team_stats,
                   umpire_data=umpire_df_hist, # Pass df with actual 'umpire' column
                   pitch_data=minimal_pitch_df
              )
              logger.info("Finished creating pitcher features.")
              if combined_features.empty: logger.error("Feature creation resulted in empty DataFrame."); sys.exit(1)
              del game_level_pitcher_stats, game_level_team_stats, umpire_df_hist, minimal_pitch_df; gc.collect()
         except Exception as e:
              logger.error(f"Error during create_pitcher_features: {e}", exc_info=True); sys.exit(1)

         logger.info("STEP 4: Final data cleanup...")
         logger.info("Final cleanup complete.")

         logger.info("STEP 5: Splitting into train/test...")
         try:
              if 'season' not in combined_features.columns: logger.error("'season' column missing."); sys.exit(1)
              combined_features['season'] = pd.to_numeric(combined_features['season'], errors='coerce')

              train_years = set(StrikeoutModelConfig.DEFAULT_TRAIN_YEARS)
              test_years = set(StrikeoutModelConfig.DEFAULT_TEST_YEARS)
              logger.info(f"Train seasons: {tuple(sorted(list(train_years)))}")
              logger.info(f"Test seasons: {tuple(sorted(list(test_years)))}")

              train_df = combined_features[combined_features['season'].isin(train_years)].copy()
              test_df = combined_features[combined_features['season'].isin(test_years)].copy()

              logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
              if train_df.empty or test_df.empty: logger.error("Train or test set empty."); sys.exit(1)
              del combined_features; gc.collect()
         except Exception as e:
              logger.error(f"Error during train/test split: {e}", exc_info=True); sys.exit(1)

         logger.info("STEP 6: Applying target encoding...")
         cols_to_encode = StrikeoutModelConfig.TARGET_ENCODING_COLS
         target_col = StrikeoutModelConfig.TARGET_VARIABLE
         try:
              if target_col not in train_df.columns: logger.error(f"Target '{target_col}' not found."); sys.exit(1)
              train_df[target_col] = pd.to_numeric(train_df[target_col], errors='coerce')
              if train_df[target_col].isnull().any():
                   logger.warning(f"NaNs found in target variable '{target_col}'. Dropping rows.")
                   train_df = train_df.dropna(subset=[target_col]) # Use assignment

              valid_cols_to_encode = [col for col in cols_to_encode if col in train_df.columns]
              if not valid_cols_to_encode:
                   logger.warning("No valid columns found for target encoding.")
                   train_features = train_df
                   test_features = test_df
              else:
                   logger.info(f" Applying target encoding to: {valid_cols_to_encode}")
                   train_features, test_features, _ = apply_target_encoding(
                        train_df, test_df, target_col, valid_cols_to_encode
                   )
              del train_df, test_df; gc.collect()
         except Exception as e:
              logger.error(f"Error during target encoding: {e}", exc_info=True); sys.exit(1)

         logger.info("STEP 7: Saving final datasets to database...")
         try:
              with DBConnection(db_path) as conn:
                   logger.info("Saving training features to 'train_features' table...")
                   train_features.to_sql('train_features', conn, if_exists='replace', index=False)
                   logger.info(f"Saved {len(train_features)} training records to 'train_features'")

                   logger.info("Saving test features to 'test_features' table...")
                   test_features.to_sql('test_features', conn, if_exists='replace', index=False)
                   logger.info(f"Saved {len(test_features)} test records to 'test_features'")
         except Exception as e:
              logger.error(f"Error saving final datasets: {e}", exc_info=True); sys.exit(1)
    # --- End Historical/Training Mode ---

    end_pipeline_time = time.time()
    logger.info(f"=== Feature Engineering Pipeline Completed in {(end_pipeline_time - start_pipeline_time):.2f}s ===")

# --- Main Execution Block ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK:
        sys.exit("Exiting: Failed module imports.")

    args = parse_args()
    logger.info("=== Feature Engineering Started ===")
    run_feature_pipeline(args)
    logger.info("=== Feature Engineering Finished Successfully ===")
