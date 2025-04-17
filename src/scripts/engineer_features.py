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
from category_encoders import TargetEncoder
from datetime import datetime, timedelta, date # Import date
import joblib # For loading/saving encoder object
import sqlite3 # For specific error handling

# Assuming script is run via python -m src.scripts.engineer_features
try:
    # Ensure config has TARGET_ENCODING_COLS updated if umpire is encoded
    from src.config import DBConfig, StrikeoutModelConfig, LogConfig, FileConfig
    from src.data.utils import setup_logger, DBConnection, find_latest_file
    from src.data.aggregate_statcast import aggregate_statcast_pitchers_sql, aggregate_statcast_batters_sql
    from src.features.pitcher_features import (
        create_pitcher_features,
        create_recency_weighted_features,
        create_trend_features,
        create_rest_features,
        create_arsenal_features
    )
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    MODULE_IMPORTS_OK = False
    sys.exit(1)

# Setup logger
LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger('engineer_features', LogConfig.LOG_DIR / 'engineer_features.log', level=logging.INFO)

# --- Constants ---
UMPIRE_COL = 'home_plate_umpire' # Use the actual column name from master_schedule
PREDICTION_EVAL_TABLE = 'prediction_evaluation'

# --- Argument Parser ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Engineer Features for MLB Strikeout Prediction Model.")
    parser.add_argument("--prediction-date", type=str, default=None,
                        help="If specified, generate features only for this date (YYYY-MM-DD) for prediction.")
    return parser.parse_args()

# --- Target Encoding Function ---
def apply_target_encoding(train_df, test_df, target_col, cols_to_encode, prediction_mode=False, encoder_path=None, placeholder='MISSING_CATEGORY'):
    """
    Applies target encoding. Fits/saves encoder in train mode, loads/applies in prediction mode.
    Keeps the original categorical columns alongside the new encoded columns.
    """
    logger.info(f"Applying target encoding (Prediction Mode: {prediction_mode})...")
    # Work on a copy of the dataframe that will be transformed (test_df)
    df_to_transform = test_df.copy()
    encoder = None
    # Identify columns requested for encoding that actually exist in the dataframe
    valid_cols_to_encode = [col for col in cols_to_encode if col in df_to_transform.columns]

    if not valid_cols_to_encode:
        logger.warning("No valid columns found in the dataframe for target encoding.")
        # Return original df and no encoder if no columns are valid
        return df_to_transform, None

    # Fill NaNs in the columns to be encoded before transforming or fitting
    for col in valid_cols_to_encode:
        if df_to_transform[col].isnull().any():
             logger.debug(f"Filling NaNs in '{col}' with placeholder '{placeholder}'.")
             # Use .loc to ensure modification happens on the DataFrame itself
             df_to_transform.loc[:, col] = df_to_transform[col].fillna(placeholder)

    # Initialize an empty DataFrame to hold the results of the encoding
    encoded_features = pd.DataFrame(index=df_to_transform.index)

    if prediction_mode:
        # --- Prediction Mode: Load Encoder and Transform ---
        if not encoder_path or not Path(encoder_path).exists():
            logger.error(f"Encoder file not found at {encoder_path}. Cannot apply encoding.")
            # Add empty encoded columns as fallback
            for col in valid_cols_to_encode:
                 df_to_transform[f"{col}_encoded"] = np.nan
            return df_to_transform, None # Return original df (with NaN encoded cols)
        try:
            logger.info(f"Loading TargetEncoder object from: {encoder_path}")
            with open(encoder_path, 'rb') as f: encoder = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load target encoder object: {e}", exc_info=True)
            for col in valid_cols_to_encode: df_to_transform[f"{col}_encoded"] = np.nan
            return df_to_transform, None

        # Transform prediction data
        try:
            logger.info(f" Applying loaded target encoding to: {valid_cols_to_encode}")
            # Ensure the encoder's columns attribute matches if needed
            if hasattr(encoder, 'cols') and encoder.cols != valid_cols_to_encode:
                 logger.warning(f"Encoder columns ({encoder.cols}) differ from requested ({valid_cols_to_encode}). Using requested.")
                 encoder.cols = valid_cols_to_encode # Align columns before transform if necessary
            encoded_features = encoder.transform(df_to_transform[valid_cols_to_encode])
        except Exception as e:
            logger.error(f"Error transforming data with loaded encoder: {e}", exc_info=True)
            # Add NaN columns as fallback if transform fails
            for col in valid_cols_to_encode: encoded_features[f"{col}_encoded"] = np.nan
            # Still concat below, will just add NaN columns

    else:
        # --- Training/Fitting Mode ---
        # Prepare training data (if provided) for fitting
        if train_df is None or train_df.empty:
             logger.error("Training dataframe (train_df) is missing or empty in fitting mode.")
             return df_to_transform, None

        train_df_copy = train_df.copy()
        # Fill NaNs in training data as well
        for col in valid_cols_to_encode:
             if col in train_df_copy.columns and train_df_copy[col].isnull().any():
                  # Use .loc for safety
                  train_df_copy.loc[:, col] = train_df_copy[col].fillna(placeholder)

        # Ensure target is numeric and handle potential NaNs before fitting
        train_df_copy[target_col] = pd.to_numeric(train_df_copy[target_col], errors='coerce')
        train_df_clean = train_df_copy.dropna(subset=[target_col] + valid_cols_to_encode).copy()
        rows_dropped = len(train_df_copy) - len(train_df_clean)
        if rows_dropped > 0:
            logger.warning(f"Dropped {rows_dropped} rows with NaN target/features before fitting encoder.")

        if train_df_clean.empty:
             logger.error("Training data is empty after handling NaNs. Cannot fit encoder.")
             # Add empty encoded columns to the output df before returning
             for col in valid_cols_to_encode: df_to_transform[f"{col}_encoded"] = np.nan
             return df_to_transform, None

        # Fit the encoder
        logger.info(f"Fitting TargetEncoder on {len(train_df_clean)} training rows...")
        min_samples_leaf = 20; smoothing = 10.0 # Consider making configurable
        encoder = TargetEncoder(cols=valid_cols_to_encode, min_samples_leaf=min_samples_leaf, smoothing=smoothing)
        try: encoder.fit(train_df_clean[valid_cols_to_encode], train_df_clean[target_col])
        except Exception as e:
             logger.error(f"Error fitting TargetEncoder: {e}", exc_info=True)
             for col in valid_cols_to_encode: df_to_transform[f"{col}_encoded"] = np.nan
             return df_to_transform, None # Return original df if fit fails

        # Save the fitted encoder
        if encoder_path:
            try:
                encoder_path.parent.mkdir(parents=True, exist_ok=True)
                with open(encoder_path, 'wb') as f: pickle.dump(encoder, f)
                logger.info(f"Saved fitted TargetEncoder object to {encoder_path}")
            except Exception as e: logger.error(f"Failed to save target encoder object: {e}")
        else: logger.warning("No encoder_path provided. Fitted encoder will not be saved.")

        # Transform the test/transform data using the newly fitted encoder
        logger.info(f"Applying newly fitted target encoding to: {valid_cols_to_encode}")
        encoded_features = encoder.transform(df_to_transform[valid_cols_to_encode])

    # --- Combine Results ---
    # Rename encoded columns
    encoded_cols_map = {col: f"{col}_encoded" for col in valid_cols_to_encode}
    encoded_features = encoded_features.rename(columns=encoded_cols_map)

    # Combine encoded features back with the original dataframe
    # **Crucially, we DO NOT drop the original columns here**
    df_out = pd.concat([df_to_transform, encoded_features], axis=1)

    return df_out, encoder

# --- Umpire Prediction Function (Unchanged from previous version) ---
def predict_home_plate_umpire(game_date_dt, home_team_abbr, away_team_abbr, schedule_df, historical_umpire_df):
    """
    Predicts the home plate umpire based on series rotation using historical data.
    (Assumes 'first_base_umpire' exists in historical_umpire_df)
    """
    try:
        if schedule_df is None or schedule_df.empty or historical_umpire_df is None or historical_umpire_df.empty:
            logger.debug("Schedule or historical umpire data missing for prediction.")
            return "Unknown"
        # Convert prediction date to date object if needed
        if not isinstance(game_date_dt, date):
             try: game_date_dt = pd.to_datetime(game_date_dt).date()
             except: logger.warning("game_date_dt is not a valid date type."); return "Unknown"

        # Ensure date columns are date objects for comparison
        schedule_df['game_date'] = pd.to_datetime(schedule_df['game_date'], errors='coerce').dt.date
        historical_umpire_df['game_date'] = pd.to_datetime(historical_umpire_df['game_date'], errors='coerce').dt.date
        schedule_df = schedule_df.dropna(subset=['game_date'])
        historical_umpire_df = historical_umpire_df.dropna(subset=['game_date'])

        lookback_days = 4
        start_series_check_dt = game_date_dt - timedelta(days=lookback_days)

        series_games = schedule_df[
            (schedule_df['game_date'] >= start_series_check_dt) &
            (schedule_df['game_date'] <= game_date_dt) &
            ( ((schedule_df['home_team'] == home_team_abbr) & (schedule_df['away_team'] == away_team_abbr)) |
              ((schedule_df['home_team'] == away_team_abbr) & (schedule_df['away_team'] == home_team_abbr)) )
        ].sort_values('game_date')

        if len(series_games) <= 1:
            logger.debug(f"Predict Umpire: First game of series or short history.")
            return "Unknown"

        previous_games_in_series = series_games[series_games['game_date'] < game_date_dt]
        if previous_games_in_series.empty:
             logger.debug(f"Predict Umpire: No previous games found in series before target date.")
             return "Unknown"

        last_game_row = previous_games_in_series.iloc[-1]
        last_game_date = last_game_row['game_date']
        last_game_home_team = last_game_row['home_team']
        last_game_away_team = last_game_row['away_team']

        if 'first_base_umpire' not in historical_umpire_df.columns:
             logger.error("Historical umpire data missing 'first_base_umpire'. Cannot predict.")
             return "Unknown"

        prev_game_umpires = historical_umpire_df[
            (historical_umpire_df['game_date'] == last_game_date) &
            (historical_umpire_df['home_team'] == last_game_home_team) &
            (historical_umpire_df['away_team'] == last_game_away_team)
        ]

        if prev_game_umpires.empty:
            logger.warning(f"Predict Umpire: Missing umpire data for previous game {last_game_date}.")
            return "Unknown"

        prev_1b_ump = prev_game_umpires['first_base_umpire'].iloc[0]

        if pd.isna(prev_1b_ump) or not prev_1b_ump:
            logger.warning(f"Predict Umpire: Previous 1B umpire name missing for {last_game_date}.")
            return "Unknown"

        logger.info(f"Predicted HP Umpire for {game_date_dt} ({away_team_abbr} @ {home_team_abbr}): {prev_1b_ump} (was 1B on {last_game_date})")
        return prev_1b_ump

    except Exception as e:
        logger.error(f"Error predicting umpire: {e}", exc_info=True)
        return "Unknown"


# --- Function to Save Umpire Predictions (Unchanged from previous version) ---
def save_umpire_predictions_to_db(predictions_df, db_path, table_name=PREDICTION_EVAL_TABLE):
    """
    Saves the predicted umpire data to the specified evaluation table.
    Creates the table if it doesn't exist. Handles updates by deleting existing
    records for the prediction date before inserting new ones.
    """
    if predictions_df.empty:
        logger.info("No umpire predictions to save.")
        return True

    required_cols = ['game_date', 'home_team', 'away_team', 'pitcher_id',
                     'predicted_strikeouts', 'predicted_home_plate_umpire']
    if not all(col in predictions_df.columns for col in required_cols):
        logger.error(f"Prediction dataframe missing required columns for saving: {required_cols}")
        return False

    try:
        # Use the first game_date (should all be the same for a prediction run)
        prediction_date_str = pd.to_datetime(predictions_df['game_date'].iloc[0]).strftime('%Y-%m-%d')
    except Exception:
        logger.error("Could not determine prediction date from dataframe.")
        return False

    logger.info(f"Saving {len(predictions_df)} umpire predictions to '{table_name}' for date {prediction_date_str}...")

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        game_date TEXT, home_team TEXT, away_team TEXT, pitcher_id INTEGER,
        predicted_strikeouts REAL, predicted_home_plate_umpire TEXT,
        prediction_timestamp TEXT,
        PRIMARY KEY (game_date, home_team, away_team, pitcher_id)
    );"""
    delete_sql = f"DELETE FROM {table_name} WHERE game_date = ?"
    insert_sql = f"""INSERT OR REPLACE INTO {table_name} (game_date, home_team, away_team, pitcher_id,
                              predicted_strikeouts, predicted_home_plate_umpire, prediction_timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?);""" # Changed to INSERT OR REPLACE

    try:
        with DBConnection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            # Delete existing records for the prediction date - Using INSERT OR REPLACE now, so deletion is optional
            # logger.debug(f"Deleting existing records for date {prediction_date_str} from {table_name}...")
            # cursor.execute(delete_sql, (prediction_date_str,))
            # logger.debug(f"Deletion complete (affected rows: {cursor.rowcount}).")

            now_timestamp = datetime.now().isoformat()
            data_to_insert = [
                (pd.to_datetime(row['game_date']).strftime('%Y-%m-%d'), row['home_team'], row['away_team'],
                 row['pitcher_id'], row['predicted_strikeouts'], row['predicted_home_plate_umpire'], now_timestamp)
                for _, row in predictions_df.iterrows()
            ]
            cursor.executemany(insert_sql, data_to_insert)
            conn.commit()
            logger.info(f"Successfully saved/updated {len(data_to_insert)} records in '{table_name}'.")
            return True
    except Exception as e:
        logger.error(f"Database error saving umpire predictions to '{table_name}': {e}", exc_info=True)
        return False

# --- Main Feature Engineering Pipeline ---
def run_feature_pipeline(args):
    """Runs the full feature engineering pipeline."""
    start_pipeline_time = time.time()
    db_path = Path(DBConfig.PATH)
    prediction_mode = args.prediction_date is not None
    model_dir = Path(FileConfig.MODELS_DIR) # Define model directory

    if prediction_mode:
        # --- PREDICTION MODE ---
        logger.info(f"=== Starting PREDICTION Feature Engineering for Date: {args.prediction_date} ===")
        output_table = "prediction_features" # Target table for base features
        try:
            prediction_date_dt = datetime.strptime(args.prediction_date, '%Y-%m-%d').date()
            max_hist_date_str = (prediction_date_dt - timedelta(days=1)).strftime('%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid prediction date format: {args.prediction_date}. Use YYYY-MM-DD."); sys.exit(1)

        logger.info("STEP 1: Loading prediction schedule and historical data...")
        pred_schedule_df = pd.DataFrame()
        master_schedule_hist_df = pd.DataFrame()
        game_level_team_stats = pd.DataFrame()
        minimal_pitch_df = pd.DataFrame()
        historical_pitcher_stats = pd.DataFrame()
        pitcher_ids_to_load = tuple()

        try:
              with DBConnection(db_path) as conn:
                   # Load prediction schedule from mlb_api
                   pred_schedule_query = f"SELECT * FROM mlb_api WHERE DATE(game_date) = '{args.prediction_date}'"
                   pred_schedule_df = pd.read_sql_query(pred_schedule_query, conn)
                   logger.info(f"Loaded {len(pred_schedule_df)} scheduled games for {args.prediction_date} from mlb_api.")
                   if pred_schedule_df.empty:
                        logger.error(f"No scheduled games found for prediction date {args.prediction_date} in 'mlb_api'. Cannot proceed.")
                        sys.exit(1)

                   # Load MASTER SCHEDULE (for umpire prediction history)
                   schedule_cols = ['game_date', 'home_team', 'away_team', UMPIRE_COL,
                                    'first_base_umpire', 'second_base_umpire', 'third_base_umpire']
                   schedule_query = f"SELECT {', '.join(schedule_cols)} FROM master_schedule WHERE DATE(game_date) < '{args.prediction_date}'" # Only need history
                   master_schedule_hist_df = pd.read_sql_query(schedule_query, conn)
                   logger.info(f"Loaded {len(master_schedule_hist_df)} historical schedule/umpire records from master_schedule.")

                   # Extract pitcher IDs from prediction schedule
                   home_pitchers = pred_schedule_df[['home_probable_pitcher_id', 'home_probable_pitcher_name']].rename(columns={'home_probable_pitcher_id':'pitcher_id', 'home_probable_pitcher_name':'player_name'})
                   away_pitchers = pred_schedule_df[['away_probable_pitcher_id', 'away_probable_pitcher_name']].rename(columns={'away_probable_pitcher_id':'pitcher_id', 'away_probable_pitcher_name':'player_name'})
                   all_pitchers = pd.concat([home_pitchers, away_pitchers]).dropna(subset=['pitcher_id']).drop_duplicates(subset=['pitcher_id'])
                   # Ensure pitcher IDs are integers before creating the tuple
                   pitcher_ids_to_load = tuple(all_pitchers['pitcher_id'].astype(int).tolist())

                   if not pitcher_ids_to_load:
                       logger.error("No valid pitcher IDs found in mlb_api data for the prediction date.")
                       sys.exit(1)
                   logger.info(f"Identified {len(pitcher_ids_to_load)} unique probable pitchers.")

                   # Load game_level_team_stats (up to yesterday)
                   team_stats_cols = ['game_pk', 'team', 'opponent', 'game_date', 'k_percent', 'bb_percent', 'swing_percent', 'contact_percent', 'swinging_strike_percent', 'chase_percent', 'zone_contact_percent']
                   team_stats_query = f"SELECT {', '.join(team_stats_cols)} FROM game_level_team_stats WHERE DATE(game_date) <= '{max_hist_date_str}'"
                   game_level_team_stats = pd.read_sql_query(team_stats_query, conn)
                   logger.info(f"Loaded {len(game_level_team_stats)} historical team stat records.")

                   # Load minimal pitch data (up to yesterday, only for needed pitchers)
                   if pitcher_ids_to_load:
                       pitch_cols = ['pitcher', 'game_pk', 'game_date', 'stand', 'events']
                       pitcher_id_list_str = ','.join(map(str, pitcher_ids_to_load))
                       pitch_query = f"SELECT {', '.join(pitch_cols)} FROM statcast_pitchers WHERE DATE(game_date) <= '{max_hist_date_str}' AND pitcher IN ({pitcher_id_list_str})"
                       minimal_pitch_df = pd.read_sql_query(pitch_query, conn)
                       logger.info(f"Loaded {len(minimal_pitch_df)} minimal pitch records for probable pitchers.")
                   else:
                       logger.warning("No probable pitchers identified, skipping minimal pitch data load.")
                       minimal_pitch_df = pd.DataFrame()


                   # Load historical game_level_pitcher data (up to yesterday, only for needed pitchers)
                   if pitcher_ids_to_load:
                       # IMPORTANT: Select 'p_throws' here if using this for handedness lookup
                       pitcher_stats_query = f"SELECT * FROM game_level_pitchers WHERE DATE(game_date) <= '{max_hist_date_str}' AND pitcher_id IN ({pitcher_id_list_str})"
                       historical_pitcher_stats = pd.read_sql_query(pitcher_stats_query, conn)
                       logger.info(f"Loaded {len(historical_pitcher_stats)} historical pitcher stat records for probable pitchers.")
                   else:
                       logger.warning("No probable pitchers identified, skipping historical pitcher stats load.")
                       historical_pitcher_stats = pd.DataFrame()

        except Exception as e:
              logger.error(f"Error loading data for prediction: {e}", exc_info=True); sys.exit(1)

        # --- Create p_throws lookup from historical data ---
        pthrows_lookup = {}
        default_pthrows = 'R' # Default to Righty if unknown
        if not historical_pitcher_stats.empty and 'pitcher_id' in historical_pitcher_stats.columns and 'p_throws' in historical_pitcher_stats.columns:
            try:
                # Get the most recent p_throws for each pitcher
                pthrows_lookup = historical_pitcher_stats.sort_values('game_date') \
                                    .drop_duplicates(subset=['pitcher_id'], keep='last') \
                                    .set_index('pitcher_id')['p_throws'] \
                                    .to_dict()
                logger.info("Created p_throws lookup from historical data.")
            except Exception as lookup_e:
                 logger.warning(f"Failed to create p_throws lookup: {lookup_e}. Using default '{default_pthrows}'.")
        else:
            logger.warning("Could not create p_throws lookup from historical_pitcher_stats (missing data or columns). Using default.")

        # --- Pre-calculate historical features ---
        logger.info("Pre-calculating historical features (lags, EWMAs, etc.) for context...")
        hist_calc_df = pd.DataFrame(columns=['pitcher_id']) # Initialize empty df with key
        if not historical_pitcher_stats.empty:
            try:
                # Define metrics needed (ensure these columns exist in historical_pitcher_stats)
                recency_metrics_all = ['strikeouts', 'batters_faced', 'innings_pitched', 'total_pitches', 'avg_velocity', 'max_velocity', 'zone_percent', 'swinging_strike_percent', 'fastball_percent', 'breaking_percent', 'offspeed_percent', 'k_percent', 'k_per_9']
                trend_metrics_all = ['strikeouts', 'innings_pitched', 'batters_faced', 'swinging_strike_percent', 'avg_velocity', 'k_percent', 'k_per_9']
                arsenal_metrics_all = ['fastball_percent', 'breaking_percent', 'offspeed_percent']

                # Filter metrics based on actual columns present
                recency_metrics = [m for m in recency_metrics_all if m in historical_pitcher_stats.columns]
                trend_metrics = [m for m in trend_metrics_all if m in historical_pitcher_stats.columns]
                arsenal_metrics = [m for m in arsenal_metrics_all if m in historical_pitcher_stats.columns]

                # Calculate features on a copy of the historical data
                temp_hist_df = historical_pitcher_stats.copy()
                temp_hist_df = create_recency_weighted_features(temp_hist_df, recency_metrics)
                temp_hist_df = create_trend_features(temp_hist_df, trend_metrics)
                temp_hist_df = create_rest_features(temp_hist_df) # Use the robust version
                temp_hist_df = create_arsenal_features(temp_hist_df, arsenal_metrics)
                hist_calc_df = temp_hist_df # Assign the result
                logger.info("Finished pre-calculating historical features.")
            except Exception as hist_e:
                logger.error(f"Error pre-calculating historical features: {hist_e}", exc_info=True)
                # Keep hist_calc_df as the initialized empty df
        else:
            logger.warning("Historical pitcher stats empty, cannot pre-calculate features.")


        logger.info("STEP 2: Constructing baseline DataFrame for prediction...")
        baseline_data = []
        default_pthrows = 'R' # Default to Righty if unknown

        for _, game in pred_schedule_df.iterrows():
             game_date_str = pd.to_datetime(game['game_date']).strftime('%Y-%m-%d')

             # --- Corrected Pitcher ID Handling & COLUMN NAMES ---
             # Use the CORRECT column names from mlb_api table schema
             home_pid_raw = pd.to_numeric(game.get('home_probable_pitcher_id'), errors='coerce') # <<< CORRECTED NAME
             away_pid_raw = pd.to_numeric(game.get('away_probable_pitcher_id'), errors='coerce') # <<< CORRECTED NAME

             home_pitcher_id = int(home_pid_raw) if pd.notna(home_pid_raw) else None
             away_pitcher_id = int(away_pid_raw) if pd.notna(away_pid_raw) else None
             # --- End Corrected Handling ---

             # Home Pitcher Row
             if home_pitcher_id is not None: # Check if ID is valid after conversion
                  baseline_data.append({
                       'pitcher_id': home_pitcher_id,
                       # Use corresponding correct name column
                       'player_name': game.get('home_probable_pitcher_name'), # <<< CORRECTED NAME
                       'game_pk': game.get('game_pk'),
                       'game_date': game_date_str,
                       'home_team': game.get('home_team'), # Assuming these are correct
                       'away_team': game.get('away_team'), # Assuming these are correct
                       'team': game.get('home_team'),
                       'opponent_team': game.get('away_team'),
                       'is_home': 1,
                       'p_throws': pthrows_lookup.get(home_pitcher_id, default_pthrows)
                  })
             # Away Pitcher Row
             if away_pitcher_id is not None: # Check if ID is valid after conversion
                  baseline_data.append({
                       'pitcher_id': away_pitcher_id,
                        # Use corresponding correct name column
                       'player_name': game.get('away_probable_pitcher_name'), # <<< CORRECTED NAME
                       'game_pk': game.get('game_pk'),
                       'game_date': game_date_str,
                       'home_team': game.get('home_team'),
                       'away_team': game.get('away_team'),
                       'team': game.get('away_team'),
                       'opponent_team': game.get('home_team'),
                       'is_home': 0,
                       'p_throws': pthrows_lookup.get(away_pitcher_id, default_pthrows)
                  })

        if not baseline_data:
             logger.error("No valid pitcher data constructed for baseline. Check pitcher IDs in mlb_api table for the prediction date.")
             sys.exit(1)

        pred_baseline_df = pd.DataFrame(baseline_data)
        # Convert pitcher_id to Int64 *after* DataFrame creation
        pred_baseline_df['pitcher_id'] = pred_baseline_df['pitcher_id'].astype('Int64')

        pred_baseline_df = pd.DataFrame(baseline_data)
        # Convert pitcher_id AFTER DataFrame creation
        pred_baseline_df['pitcher_id'] = pred_baseline_df['pitcher_id'].astype('Int64')
        # <<< ADD THIS LINE: Convert game_pk >>>
        pred_baseline_df['game_pk'] = pd.to_numeric(pred_baseline_df['game_pk'], errors='coerce').astype('Int64')

        if pred_baseline_df.empty:
             logger.error("Constructed baseline DataFrame is empty after DataFrame creation. Check input data.")
             sys.exit(1)
        logger.info(f"Constructed baseline with {len(pred_baseline_df)} pitcher-game rows for prediction.")


        logger.info("STEP 3a: Predicting umpires and saving predictions...")
        # Prepare historical umpire data (already loaded as master_schedule_hist_df)
        historical_umpire_pred_df = master_schedule_hist_df.copy()
        # Prepare schedule subset for prediction function
        schedule_pred_df = master_schedule_hist_df[['game_date', 'home_team', 'away_team']].copy()

        predicted_umpires_list = []
        umpire_predictions_for_eval = []

        games_to_predict_umps_for = pred_baseline_df[['game_date', 'home_team', 'away_team']].drop_duplicates()

        if historical_umpire_pred_df.empty or 'first_base_umpire' not in historical_umpire_pred_df.columns:
             logger.warning("Insufficient historical umpire data in master_schedule to predict umpires. Predicting 'Unknown'.")
             for _, game_info in games_to_predict_umps_for.iterrows():
                 predicted_umpires_list.append({
                     'game_date': game_info['game_date'], 'home_team': game_info['home_team'],
                     'away_team': game_info['away_team'], UMPIRE_COL: 'Unknown'
                 })
        else:
            for _, game_info in games_to_predict_umps_for.iterrows():
                game_date_for_pred = pd.to_datetime(game_info['game_date']).date()
                pred_ump = predict_home_plate_umpire(
                    game_date_for_pred, game_info['home_team'], game_info['away_team'],
                    schedule_pred_df, historical_umpire_pred_df
                )
                predicted_umpires_list.append({
                    'game_date': game_info['game_date'], # Keep string format
                    'home_team': game_info['home_team'],
                    'away_team': game_info['away_team'],
                    UMPIRE_COL: pred_ump
                })

        umpire_df_pred = pd.DataFrame(predicted_umpires_list)
        logger.info(f"Predicted umpires for {len(umpire_df_pred)} unique games.")

        # Merge predicted umpires onto the baseline pitcher df for eval table saving
        pred_baseline_for_eval = pd.merge(
            pred_baseline_df[['game_date', 'home_team', 'away_team', 'pitcher_id']],
            umpire_df_pred,
            on=['game_date', 'home_team', 'away_team'], how='left'
        )
        pred_baseline_for_eval[UMPIRE_COL] = pred_baseline_for_eval[UMPIRE_COL].fillna('Unknown')

        # Prepare and save data for evaluation table
        eval_df_to_save = pred_baseline_for_eval.rename(columns={UMPIRE_COL: 'predicted_home_plate_umpire'}) # Rename for eval table
        eval_df_to_save['predicted_strikeouts'] = np.nan # Add placeholder
        save_success = save_umpire_predictions_to_db(eval_df_to_save, db_path)
        if not save_success: logger.warning("Failed to save umpire predictions to evaluation table.")


        logger.info("STEP 3b: Creating pitcher features for prediction date...")
        # Merge predicted umpires into the main baseline df for feature creation
        pred_baseline_merged = pd.merge(
            pred_baseline_df, umpire_df_pred,
            on=['game_date', 'home_team', 'away_team'], how='left'
        )
        pred_baseline_merged[UMPIRE_COL] = pred_baseline_merged[UMPIRE_COL].fillna('Unknown')

        # !!! Use the MODIFIED create_pitcher_features call !!!
        final_pred_features = create_pitcher_features(
              pitcher_data=pred_baseline_merged,         # Pass ONLY the prediction baseline rows
              historical_pitcher_stats=hist_calc_df,     # Pass the pre-calculated historical data
              team_stats_data=game_level_team_stats,     # Pass historical team stats
              umpire_data=umpire_df_pred,                # Pass predicted umpires (needed again inside create_umpire_features)
              pitch_data=minimal_pitch_df,               # Pass historical pitch data (for platoon)
              prediction_mode=True                       # Set prediction mode flag
         )
         # !!! End MODIFIED CALL !!!

        # Check the result immediately
        if final_pred_features.empty:
             logger.error("Feature generation returned empty dataframe after create_pitcher_features. Exiting.")
             sys.exit(1)
        required_cols_after_features = ['game_date', 'pitcher_id', 'game_pk'] # Add other critical columns
        missing_cols = [col for col in required_cols_after_features if col not in final_pred_features.columns]
        if missing_cols:
            logger.error(f"Feature generation dataframe is missing required columns after create_pitcher_features: {missing_cols}. Exiting.")
            logger.error(f"Columns present: {final_pred_features.columns.tolist()}")
            sys.exit(1)

        # Ensure we only have rows for the prediction date (should already be the case, but safety check)
        final_pred_features['game_date'] = pd.to_datetime(final_pred_features['game_date']).dt.strftime('%Y-%m-%d')
        final_pred_features = final_pred_features[
              final_pred_features['game_date'] == args.prediction_date
         ].copy()
        logger.info(f"Verified features are for prediction date. Rows: {len(final_pred_features)}")

        if final_pred_features.empty:
             logger.error("Feature dataframe became empty after filtering for prediction date.")
             sys.exit(1)


        logger.info("STEP 4: Applying target encoding using saved encoder...")
        try:
             encoder_pattern = "target_encoder_*.pkl"
             latest_encoder_path = find_latest_file(model_dir, encoder_pattern)

             if not latest_encoder_path:
                  logger.error("Could not find target encoder .pkl file. Cannot apply encoding for prediction.")
                  # Add NaN encoded columns as fallback
                  cols_to_encode = StrikeoutModelConfig.TARGET_ENCODING_COLS
                  for col in cols_to_encode:
                       if col in final_pred_features.columns:
                           final_pred_features[f"{col}_encoded"] = np.nan
                           # Drop original column if it exists after adding encoded NaN
                           final_pred_features = final_pred_features.drop(columns=[col], errors='ignore')
             else:
                cols_to_encode = StrikeoutModelConfig.TARGET_ENCODING_COLS
                logger.info(f"Columns configured for target encoding: {cols_to_encode}")
                # Apply encoding - function drops original cols in prediction mode
                final_pred_features, _ = apply_target_encoding(
                    train_df=None, test_df=final_pred_features,
                    target_col=StrikeoutModelConfig.TARGET_VARIABLE,
                    cols_to_encode=cols_to_encode, prediction_mode=True,
                    encoder_path=latest_encoder_path
                )
                # Verify original columns were dropped (optional)
                dropped_cols_check = [col for col in cols_to_encode if col in final_pred_features.columns]
                if dropped_cols_check:
                     logger.warning(f"Original target encoding columns were not dropped after encoding: {dropped_cols_check}")

        except Exception as e:
              logger.error(f"Error applying target encoding during prediction: {e}", exc_info=True); sys.exit(1)

        logger.info("STEP 5: Saving final prediction features (with encoded umpire)...")
        try:
              if UMPIRE_COL in final_pred_features.columns:
                   logger.warning(f"Raw predicted umpire column '{UMPIRE_COL}' still present before saving prediction features. Dropping.")
                   final_pred_features = final_pred_features.drop(columns=[UMPIRE_COL], errors='ignore')
              nan_inf_check = final_pred_features.isnull().sum().sum() + np.isinf(final_pred_features.select_dtypes(include=np.number)).sum().sum()
              if nan_inf_check > 0: logger.warning(f"Found {nan_inf_check} NaN/Inf values before saving.")

              with DBConnection(db_path) as conn:
                   final_pred_features.to_sql(output_table, conn, if_exists='replace', index=False)
              logger.info(f"Saved {len(final_pred_features)} prediction records to '{output_table}'")
        except Exception as e:
            logger.error(f"Error saving prediction features: {e}", exc_info=True); sys.exit(1)
         # --- END PREDICTION MODE ---

    else:
         # --- TRAIN/TEST MODE (Largely unchanged) ---
        logger.info("=== Starting Historical Feature Engineering Pipeline ===")
        logger.info("Running aggregation and feature engineering steps...")
        game_level_pitcher_stats = pd.DataFrame()
        game_level_team_stats = pd.DataFrame()
        minimal_pitch_df = pd.DataFrame()
        umpire_df_hist = pd.DataFrame()
        train_features = pd.DataFrame()
        test_features = pd.DataFrame()

        logger.info("STEP 1: Aggregating raw Statcast data...")
        try:
              game_level_pitcher_stats = aggregate_statcast_pitchers_sql()
              if game_level_pitcher_stats is None or game_level_pitcher_stats.empty: raise ValueError("Pitcher agg failed.")
              logger.info(f"Loaded/Aggregated {len(game_level_pitcher_stats)} pitcher game records.")
              game_level_team_stats = aggregate_statcast_batters_sql()
              if game_level_team_stats is None or game_level_team_stats.empty: raise ValueError("Team agg failed.")
              logger.info(f"Loaded/Aggregated {len(game_level_team_stats)} team game records.")

              if True: # Assume platoon needed
                   with DBConnection(db_path) as conn:
                        pitch_cols = ['pitcher', 'game_pk', 'game_date', 'stand', 'events']
                        pitch_query = f"SELECT {', '.join(pitch_cols)} FROM statcast_pitchers"
                        minimal_pitch_df = pd.read_sql_query(pitch_query, conn)
                        logger.info(f"Loaded {len(minimal_pitch_df)} minimal pitch records.")

              logger.info("Loading historical umpire data from master_schedule...")
              with DBConnection(db_path) as conn:
                   umpire_cols = ['game_date', 'home_team', 'away_team', UMPIRE_COL]
                   umpire_query = f"SELECT {', '.join(umpire_cols)} FROM master_schedule"
                   umpire_df_hist = pd.read_sql_query(umpire_query, conn)
                   umpire_df_hist = umpire_df_hist.dropna(subset=[UMPIRE_COL])
                   umpire_df_hist['game_date'] = pd.to_datetime(umpire_df_hist['game_date']).dt.strftime('%Y-%m-%d')
                   logger.info(f"Loaded and prepared {len(umpire_df_hist)} historical umpire records.")
        except Exception as e: logger.error(f"Error during data loading/aggregation: {e}", exc_info=True); sys.exit(1)

        logger.info("STEP 2: Creating pitcher features...")
        try:
            # Pass the combined dataframe directly (it's all historical here)
              combined_features = create_pitcher_features(
                   pitcher_data=game_level_pitcher_stats, # Pass all historical baseline data
                   historical_pitcher_stats=None,        # Not needed in training mode
                   team_stats_data=game_level_team_stats,
                   umpire_data=umpire_df_hist,           # Pass actual historical umpires
                   pitch_data=minimal_pitch_df,
                   prediction_mode=False                 # Explicitly set train mode
              )
              if combined_features.empty: logger.error("Feature creation empty."); sys.exit(1)
              # Check for essential columns after feature creation
              required_cols_after_features = ['game_date', 'pitcher_id', 'game_pk', 'season']
              missing_cols = [col for col in required_cols_after_features if col not in combined_features.columns]
              if missing_cols:
                  logger.error(f"Feature creation dataframe missing required columns: {missing_cols}. Exiting.")
                  logger.error(f"Columns present: {combined_features.columns.tolist()}")
                  sys.exit(1)
              del game_level_pitcher_stats, game_level_team_stats, umpire_df_hist, minimal_pitch_df; gc.collect()
        except Exception as e: logger.error(f"Error during create_pitcher_features: {e}", exc_info=True); sys.exit(1)

        logger.info("STEP 3: Splitting into train/test...")
        try:
              combined_features['season'] = pd.to_numeric(combined_features['season'], errors='coerce')
              train_years = set(StrikeoutModelConfig.DEFAULT_TRAIN_YEARS)
              test_years = set(StrikeoutModelConfig.DEFAULT_TEST_YEARS)
              train_df = combined_features[combined_features['season'].isin(train_years)].copy()
              test_df = combined_features[combined_features['season'].isin(test_years)].copy()
              if train_df.empty or test_df.empty: logger.error("Train or test set empty after split."); sys.exit(1)
              logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
              del combined_features; gc.collect()
        except Exception as e: logger.error(f"Error during train/test split: {e}", exc_info=True); sys.exit(1)

        logger.info("STEP 4: Applying target encoding...")
        cols_to_encode = StrikeoutModelConfig.TARGET_ENCODING_COLS
        target_col = StrikeoutModelConfig.TARGET_VARIABLE
        try:
              if target_col not in train_df.columns: logger.error(f"Target '{target_col}' not found."); sys.exit(1)
              train_df[target_col] = pd.to_numeric(train_df[target_col], errors='coerce')
              if train_df[target_col].isnull().any():
                   logger.warning(f"NaNs found in target variable. Dropping rows before fitting encoder.")
                   train_df = train_df.dropna(subset=[target_col]) # Assign back

              timestamp = datetime.now().strftime("%Y%m%d")
              encoder_save_path = model_dir / f"target_encoder_{timestamp}.pkl"

              # Fit encoder on train_df, transform test_df, and save encoder
              test_features, fitted_encoder = apply_target_encoding(
                   train_df=train_df, test_df=test_df, target_col=target_col,
                   cols_to_encode=cols_to_encode, prediction_mode=False,
                   encoder_path=encoder_save_path
              )

              # Transform train_df using the *same* fitted encoder
              if fitted_encoder:
                   # Pass train_df as the df to transform, use prediction_mode=True to load/apply
                   train_features, _ = apply_target_encoding(
                       train_df=None, test_df=train_df, target_col=target_col,
                       cols_to_encode=cols_to_encode, prediction_mode=True,
                       encoder_path=encoder_save_path
                   )
              else:
                   logger.error("Encoder fitting failed. Cannot transform training data.")
                   train_features = train_df.copy() # Fallback
                   for col in cols_to_encode:
                       if col in train_features.columns: train_features[f"{col}_encoded"] = np.nan

              del train_df, test_df; gc.collect() # Clean up original dfs
        except Exception as e: logger.error(f"Error during target encoding: {e}", exc_info=True); sys.exit(1)

        logger.info("STEP 5: Saving final datasets...")
        try:
              for name, df_save in [('train_features', train_features), ('test_features', test_features)]:
                   nan_inf_check = df_save.isnull().sum().sum() + np.isinf(df_save.select_dtypes(include=np.number)).sum().sum()
                   if nan_inf_check > 0: logger.warning(f"Found {nan_inf_check} NaN/Inf values in final {name} before saving.")

              with DBConnection(db_path) as conn:
                   train_features.to_sql('train_features', conn, if_exists='replace', index=False)
                   logger.info(f"Saved {len(train_features)} training records to 'train_features'")
                   test_features.to_sql('test_features', conn, if_exists='replace', index=False)
                   logger.info(f"Saved {len(test_features)} test records to 'test_features'")
        except Exception as e: logger.error(f"Error saving final datasets: {e}", exc_info=True); sys.exit(1)
         # --- END TRAIN/TEST MODE ---

    end_pipeline_time = time.time()
    logger.info(f"=== Feature Engineering Pipeline Completed in {(end_pipeline_time - start_pipeline_time):.2f}s ===")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure necessary feature functions are imported or defined above run_feature_pipeline
    # (Example: Make sure create_recency_weighted_features, etc. are available)
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    args = parse_args()
    if args.prediction_date:
        try: datetime.strptime(args.prediction_date, '%Y-%m-%d')
        except ValueError: logger.error(f"Invalid date format: {args.prediction_date}."); sys.exit(1)

    logger.info("=== Feature Engineering Script Started ===")
    run_feature_pipeline(args)
    logger.info("=== Feature Engineering Script Finished Successfully ===")