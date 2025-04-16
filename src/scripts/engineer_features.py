# src/scripts/engineer_features.py (Refactored - Load Minimal Pitch Data for Platoon)
import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import time
import argparse
import pickle
import gc
from datetime import datetime, timedelta
import traceback
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Direct imports
from src.data.utils import setup_logger, DBConnection
# Import the NEW SQL-based aggregation functions
from src.data.aggregate_statcast import aggregate_statcast_pitchers_sql, aggregate_statcast_batters_sql
# Import feature creation functions (ensure create_pitcher_features is the main entry point)
from src.features.pitcher_features import create_pitcher_features
# Import config separately if needed, or access via StrikeoutModelConfig
from src.config import StrikeoutModelConfig, DBConfig

# Setup logger
logger = setup_logger('engineer_features')

# Create checkpoint directory
checkpoint_dir = project_root / 'data' / 'checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)


def load_team_batting_data():
    """Load team batting data from database."""
    logger.info("Loading team batting data (e.g., from team_batting table)...")
    try:
        with DBConnection() as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            query = "SELECT * FROM team_batting"
            df = pd.read_sql_query(query, conn)
            logger.info(f"Loaded {len(df)} rows of team batting data")
            return df
    except Exception as e:
        logger.warning(f"Could not load team batting data from 'team_batting': {e}. Continuing without it if possible.")
        return pd.DataFrame()

def load_umpire_data():
    """Load umpire assignment data from database."""
    logger.info("Loading umpire data...")
    umpire_df = pd.DataFrame()
    try:
        with DBConnection() as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            query = "SELECT * FROM umpire_data"
            umpire_df = pd.read_sql_query(query, conn)
            logger.info(f"Loaded {len(umpire_df)} rows of umpire data")
            if not umpire_df.empty:
                 umpire_df['game_date'] = pd.to_datetime(umpire_df['game_date']).dt.normalize()
            return umpire_df
    except Exception as e:
        if "no such table: umpire_data" in str(e): logger.error(f"Error loading umpire data: Table 'umpire_data' not found.", exc_info=False)
        else: logger.error(f"Error loading umpire data: {e}", exc_info=True)
        return pd.DataFrame()

# *** NEW FUNCTION to load minimal pitch data for platoon features ***
def load_minimal_pitch_data():
    """Load minimal columns from raw pitch data needed for platoon features."""
    logger.info("Loading minimal pitch data for platoon features...")
    # Define required columns
    cols = ['pitcher_id', 'game_pk', 'game_date', 'at_bat_number', 'pitch_number', 'stand', 'events']
    cols_str = ", ".join(f'"{c}"' for c in cols) # Quote column names if needed, adjust based on DB dialect

    # Use the same source table as your aggregation query
    # ** IMPORTANT: Adjust table name if necessary **
    source_table = "statcast_pitchers"
    query = f"SELECT {cols_str} FROM {source_table} WHERE game_type = 'R'" # Filter regular season

    try:
        with DBConnection() as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            df = pd.read_sql_query(query, conn)
            logger.info(f"Loaded {len(df)} rows of minimal pitch data from '{source_table}'")
            if not df.empty:
                 # Ensure date type is consistent
                 df['game_date'] = pd.to_datetime(df['game_date']).dt.normalize()
            return df
    except Exception as e:
        logger.error(f"Error loading minimal pitch data: {e}", exc_info=True)
        return pd.DataFrame()


def run_historical_feature_pipeline(args):
    """
    Run the complete historical feature engineering pipeline using SQL aggregations.
    Loads minimal pitch data separately for platoon features.
    Includes expanded target encoding.

    Args:
        args: Command-line arguments

    Returns:
        bool: Success status
    """
    logger.info("=== Starting Historical Feature Engineering Pipeline (SQL Aggregation) ===")
    start_time = time.time()

    combined_checkpoint = checkpoint_dir / 'combined_features_checkpoint_sql.pkl'
    combined_df = pd.DataFrame()
    use_checkpoint = not args.ignore_checkpoint

    # Check for the FINAL combined features checkpoint
    if use_checkpoint and combined_checkpoint.exists():
        try:
            logger.info(f"Attempting to load final combined features from checkpoint: {combined_checkpoint}")
            with open(combined_checkpoint, 'rb') as f:
                combined_df = pickle.load(f)
            if isinstance(combined_df, pd.DataFrame) and not combined_df.empty and \
               'strikeouts' in combined_df.columns and 'pitcher_id' in combined_df.columns:
                logger.info(f"Loaded {len(combined_df)} final combined rows from checkpoint. Skipping aggregation and feature creation.")
            else:
                logger.warning("Combined checkpoint invalid or missing required columns. Rebuilding features.")
                combined_df = pd.DataFrame()
                use_checkpoint = False
        except Exception as e:
            logger.error(f"Failed to load combined checkpoint: {e}. Rebuilding features.")
            use_checkpoint = False
    else:
        logger.info("No valid combined features checkpoint found. Running full pipeline...")
        use_checkpoint = False

    if combined_df.empty:
        logger.info("Running aggregation and feature engineering steps...")

        # STEP 1: Aggregate raw data using SQL functions
        logger.info("STEP 1: Aggregating raw Statcast data using SQL functions...")
        game_level_pitchers = aggregate_statcast_pitchers_sql(
            use_checkpoint=(not args.ignore_checkpoint),
            force_reprocess=args.ignore_checkpoint
        )
        if game_level_pitchers.empty: return False # Error logged in function
        logger.info(f"Successfully aggregated/loaded {len(game_level_pitchers)} pitcher game records.")
        logger.info(f"Columns returned by aggregate_statcast_pitchers_sql: {game_level_pitchers.columns.tolist()}")
        if 'is_home' not in game_level_pitchers.columns: logger.error("FATAL: 'is_home' column IS MISSING immediately after aggregation!")
        else: logger.info(f"'is_home' column is present. Value counts:\n{game_level_pitchers['is_home'].value_counts(dropna=False)}")

        game_level_teams = aggregate_statcast_batters_sql(
            use_checkpoint=(not args.ignore_checkpoint),
            force_reprocess=args.ignore_checkpoint
        )
        if game_level_teams.empty: return False # Error logged in function
        logger.info(f"Successfully aggregated/loaded {len(game_level_teams)} team game records.")

        # STEP 2: Load supplementary & Platoon data
        logger.info("STEP 2: Loading supplementary data (Team Batting, Umpires, Minimal Pitch)...")
        team_batting_supp = load_team_batting_data()
        umpire_data = load_umpire_data()
        if umpire_data.empty: logger.warning("Umpire data is empty or failed to load. Umpire features cannot be created.")
        # *** Load minimal pitch data needed for platoon features ***
        minimal_pitch_data = load_minimal_pitch_data()
        if minimal_pitch_data.empty: logger.warning("Minimal pitch data failed to load. Platoon features cannot be created.")


        # STEP 3: Create pitcher features using aggregated data + minimal pitch data
        logger.info("STEP 3: Creating pitcher features...")
        pitcher_features_checkpoint = checkpoint_dir / 'pitcher_features_sql.pkl'
        pitcher_features = pd.DataFrame()

        if use_checkpoint and pitcher_features_checkpoint.exists():
             try:
                  logger.info(f"Loading pitcher features from checkpoint: {pitcher_features_checkpoint}")
                  with open(pitcher_features_checkpoint, 'rb') as f: pitcher_features = pickle.load(f)
                  if isinstance(pitcher_features, pd.DataFrame) and not pitcher_features.empty: logger.info(f"Loaded {len(pitcher_features)} pitcher features from checkpoint")
                  else: logger.warning("Invalid pitcher features checkpoint. Recreating."); pitcher_features = pd.DataFrame()
             except Exception as e: logger.warning(f"Failed to load pitcher features checkpoint: {e}. Recreating."); pitcher_features = pd.DataFrame()

        if pitcher_features.empty:
            try:
                logger.info(f"Columns passed to create_pitcher_features: {game_level_pitchers.columns.tolist()}")
                if 'is_home' not in game_level_pitchers.columns:
                     logger.error("Cannot proceed to create_pitcher_features without 'is_home' column.")
                     return False

                # *** Pass the loaded minimal_pitch_data ***
                pitcher_features = create_pitcher_features(
                    game_level_pitchers, # Pass main dataframe positionally
                    pitch_level_data=minimal_pitch_data, # Pass minimal pitch data
                    team_batting_data=game_level_teams,
                    umpire_data=umpire_data
                )
            except Exception as e:
                 logger.error(f"Error calling create_pitcher_features: {e}", exc_info=True)
                 pitcher_features = pd.DataFrame()

            if pitcher_features.empty:
                logger.error("Failed to create pitcher features. Exiting.")
                del game_level_pitchers, game_level_teams, team_batting_supp, umpire_data, minimal_pitch_data; gc.collect()
                return False

            try:
                logger.info(f"Saving pitcher features checkpoint: {pitcher_features_checkpoint}")
                with open(pitcher_features_checkpoint, 'wb') as f: pickle.dump(pitcher_features, f)
                logger.info("Pitcher features checkpoint saved")
            except Exception as e: logger.error(f"Failed to save pitcher features checkpoint: {e}")

        # Clean up large intermediate dataframes
        del game_level_pitchers, game_level_teams, team_batting_supp, umpire_data, minimal_pitch_data; gc.collect()

        # STEP 4: Final data cleanup
        logger.info("STEP 4: Final data cleanup...")
        for col in pitcher_features.select_dtypes(include=np.number).columns:
            if pitcher_features[col].isnull().sum() > 0:
                median_val = pitcher_features[col].median()
                fill_val = median_val if pd.notna(median_val) else 0
                # logger.info(f"Filling {pitcher_features[col].isnull().sum()} missing values in {col} with {fill_val:.4f}") # Reduce log verbosity
                pitcher_features[col] = pitcher_features[col].fillna(fill_val)
        logger.info("Final cleanup complete.")


        combined_df = pitcher_features.copy()

        try:
            logger.info(f"Saving combined features checkpoint: {combined_checkpoint}")
            with open(combined_checkpoint, 'wb') as f: pickle.dump(combined_df, f)
            logger.info("Combined features checkpoint saved")
        except Exception as e: logger.error(f"Failed to save combined features checkpoint: {e}")

        del pitcher_features; gc.collect()

    # --- Steps from here use the combined_df (either loaded or created) ---

    if combined_df.empty:
         logger.error("Combined features DataFrame is empty before train/test split. Exiting.")
         return False

    # STEP 5: Split into train/test
    # (Code remains the same)
    logger.info("STEP 5: Splitting into train/test...")
    if 'season' not in combined_df.columns: logger.error("Missing 'season' column"); return False
    TARGET_VARIABLE = 'strikeouts'
    if TARGET_VARIABLE not in combined_df.columns: logger.error(f"Target '{TARGET_VARIABLE}' not found"); return False
    train_seasons = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
    test_seasons = StrikeoutModelConfig.DEFAULT_TEST_YEARS
    logger.info(f"Train seasons: {train_seasons}")
    logger.info(f"Test seasons: {test_seasons}")
    combined_df['season'] = pd.to_numeric(combined_df['season'], errors='coerce')
    combined_df = combined_df.dropna(subset=['season'])
    combined_df['season'] = combined_df['season'].astype(int)
    train_df = combined_df[combined_df['season'].isin(train_seasons)].copy()
    test_df = combined_df[combined_df['season'].isin(test_seasons)].copy()
    logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")
    if train_df.empty: logger.error("Empty training set"); return False


    # STEP 6: Apply target encoding for multiple categorical features
    # (Code remains the same)
    logger.info("STEP 6: Applying target encoding...")
    columns_to_encode = ['umpire', 'home_team', 'opponent_team', 'p_throws']
    encoding_map = {}
    global_mean = train_df[TARGET_VARIABLE].mean()
    logger.info(f"Global mean {TARGET_VARIABLE}: {global_mean:.4f}")
    for column in columns_to_encode:
        logger.info(f"  Target encoding column: {column}")
        if column in train_df.columns:
            encoding_map_col = train_df.groupby(column)[TARGET_VARIABLE].mean()
            train_df[f'{column}_encoded'] = train_df[column].map(encoding_map_col)
            if not test_df.empty and column in test_df.columns: test_df[f'{column}_encoded'] = test_df[column].map(encoding_map_col)
            train_df[f'{column}_encoded'] = train_df[f'{column}_encoded'].fillna(global_mean)
            if not test_df.empty and f'{column}_encoded' in test_df.columns: test_df[f'{column}_encoded'] = test_df[f'{column}_encoded'].fillna(global_mean)
            encoding_map[column] = {'map': encoding_map_col.to_dict(), 'default': global_mean}
            logger.info(f"    Encoded {len(encoding_map_col)} categories for {column}.")
        else: logger.warning(f"    Column '{column}' not found in training data. Skipping target encoding for it.")
    if encoding_map:
        try:
            output_dir = project_root / 'models'; output_dir.mkdir(exist_ok=True, parents=True)
            map_file = output_dir / f'encoding_map_{datetime.now().strftime("%Y%m%d")}.pkl'
            with open(map_file, 'wb') as f: pickle.dump(encoding_map, f)
            logger.info(f"Saved consolidated encoding map to {map_file}")
        except Exception as e: logger.error(f"Failed to save encoding map: {e}")
    else: logger.warning("No columns were target encoded.")


    # STEP 7: Save final train/test datasets to Database
    # (Code remains the same)
    logger.info("STEP 7: Saving final datasets to database...")
    try:
        with DBConnection() as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            train_table = 'train_features'; test_table = 'test_features'
            logger.info(f"Saving training features to '{train_table}' table...")
            train_df.to_sql(train_table, conn, if_exists='replace', index=False, chunksize=10000)
            logger.info(f"Saved {len(train_df)} training records to '{train_table}'")
            if not test_df.empty:
                logger.info(f"Saving test features to '{test_table}' table...")
                test_df.to_sql(test_table, conn, if_exists='replace', index=False, chunksize=10000)
                logger.info(f"Saved {len(test_df)} test records to '{test_table}'")
            else: logger.info("Test set is empty. Skipping save.")
    except Exception as e: logger.error(f"Failed to save final datasets to database: {e}", exc_info=True); return False


    del train_df, test_df, combined_df; gc.collect()
    total_time = time.time() - start_time
    logger.info(f"=== Historical Feature Engineering Pipeline Completed in {total_time:.2f}s ===")
    return True


# Update function definition to accept args
def generate_prediction_features(prediction_date_str, args):
    """
    Generate features for prediction on a specific date using game-level data.
    Applies multiple target encodings based on saved map.
    Note: Does not load minimal pitch data for platoon features in prediction.

    Args:
        prediction_date_str: Date string in YYYY-MM-DD format
        args: Command-line arguments object (contains --ignore-checkpoint)

    Returns:
        bool: Success status
    """
    logger.info(f"=== Generating Prediction Features for {prediction_date_str} ===")
    start_time = time.time()
    use_checkpoint = not args.ignore_checkpoint # Determine if checkpoint should be used

    prediction_checkpoint = checkpoint_dir / f'prediction_features_{prediction_date_str}_sql.pkl'

    # *** MODIFIED CHECKPOINT LOGIC ***
    if use_checkpoint and prediction_checkpoint.exists(): # Only check if use_checkpoint is True
        try:
            logger.info(f"Attempting to load from prediction checkpoint: {prediction_checkpoint}")
            with open(prediction_checkpoint, 'rb') as f: prediction_features = pickle.load(f)
            if isinstance(prediction_features, pd.DataFrame) and not prediction_features.empty:
                logger.info(f"Loaded {len(prediction_features)} prediction records from checkpoint")
                try: # Optional: Save to DB
                    with DBConnection() as conn:
                        if conn is not None: prediction_features.to_sql('prediction_features', conn, if_exists='replace', index=False); logger.info(f"Updated prediction_features table from checkpoint")
                except Exception as e: logger.warning(f"Failed to update prediction table from checkpoint: {e}")
                # Successfully loaded from checkpoint and updated DB, task is done
                logger.info("=== Feature Engineering Finished Successfully (Loaded from Checkpoint) ===")
                return True # Exit successfully after loading checkpoint
            else:
                logger.warning("Invalid or empty prediction checkpoint found. Regenerating features.")
        except Exception as e:
            logger.warning(f"Failed to load prediction checkpoint: {e}. Regenerating features.")
    elif not use_checkpoint:
         logger.info("Ignoring existing checkpoint as per --ignore-checkpoint flag.")
    else: # No checkpoint exists
         logger.info("No prediction checkpoint found. Generating features...")
    # *** END MODIFIED CHECKPOINT LOGIC ***


    # Parse date
    # (Code remains the same)
    try:
        prediction_date = datetime.strptime(prediction_date_str, "%Y-%m-%d").date()
        prediction_dt = datetime.combine(prediction_date, datetime.min.time())
    except ValueError: logger.error(f"Invalid date format: {prediction_date_str}. Use YYYY-MM-DD."); return False


    # STEP 1: Load LATEST historical GAME-LEVEL data
    # (Code remains the same)
    logger.info("STEP 1: Loading historical game-level pitcher data...")
    try:
        with DBConnection() as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            query = f"SELECT * FROM game_level_pitchers WHERE game_date < '{prediction_date_str}'"
            historical_df = pd.read_sql_query(query, conn)
            logger.info(f"Loaded {len(historical_df)} historical game records from 'game_level_pitchers'")
            if historical_df.empty: logger.error("No historical pitcher game-level data found."); return False
            historical_df['game_date'] = pd.to_datetime(historical_df['game_date']).dt.normalize()
            if 'is_home' not in historical_df.columns: logger.error("Historical data missing 'is_home' column!"); return False
    except Exception as e: logger.error(f"Failed to load historical game-level data: {e}", exc_info=True); return False


    # STEP 2: Load scheduled game/pitcher data for target date
    # (Code remains the same)
    logger.info("STEP 2: Loading scheduled games for prediction date...")
    try:
        with DBConnection() as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            query = f"SELECT * FROM mlb_api WHERE game_date = '{prediction_date_str}'"
            scheduled_games = pd.read_sql_query(query, conn)
            if scheduled_games.empty: logger.error(f"No scheduled games found for {prediction_date_str} in 'mlb_api'."); return False
            logger.info(f"Loaded {len(scheduled_games)} scheduled games")
    except Exception as e: logger.error(f"Failed to load scheduled games: {e}", exc_info=True); return False


    # STEP 3: Load supplementary data needed for feature creation
    # (Code remains the same)
    logger.info("STEP 3: Loading supplementary data (Team Batting Aggs, Umpires)...")
    team_game_stats = pd.DataFrame()
    try:
         with DBConnection() as conn:
              if conn is None: raise ConnectionError("DB Connection failed.")
              query_teams = f"SELECT * FROM game_level_team_stats WHERE game_date < '{prediction_date_str}'"
              team_game_stats = pd.read_sql_query(query_teams, conn)
              logger.info(f"Loaded {len(team_game_stats)} historical team game stats")
              if not team_game_stats.empty: team_game_stats['game_date'] = pd.to_datetime(team_game_stats['game_date']).dt.normalize()
    except Exception as e: logger.error(f"Failed to load historical team game stats: {e}", exc_info=True)
    umpire_data = load_umpire_data()


    # STEP 4: Create prediction records shell
    # (Code remains the same - already includes is_home)
    logger.info("STEP 4: Creating prediction records shell...")
    prediction_records = []
    for _, game in scheduled_games.iterrows():
        home_pitcher_id = pd.to_numeric(game.get('home_probable_pitcher_id'), errors='coerce')
        away_pitcher_id = pd.to_numeric(game.get('away_probable_pitcher_id'), errors='coerce')
        if pd.notna(home_pitcher_id): prediction_records.append({'pitcher_id': int(home_pitcher_id),'player_name': game.get('home_probable_pitcher_name'),'game_pk': game['gamePk'],'game_date': prediction_dt,'home_team': game['home_team_abbr'],'away_team': game['away_team_abbr'],'is_home': 1,'season': prediction_dt.year})
        if pd.notna(away_pitcher_id): prediction_records.append({'pitcher_id': int(away_pitcher_id),'player_name': game.get('away_probable_pitcher_name'),'game_pk': game['gamePk'],'game_date': prediction_dt,'home_team': game['home_team_abbr'],'away_team': game['away_team_abbr'],'is_home': 0,'season': prediction_dt.year})
    if not prediction_records: logger.error("No valid probable pitchers found."); return False
    prediction_df = pd.DataFrame(prediction_records)
    logger.info(f"Created {len(prediction_df)} prediction shells.")


    # STEP 5: Combine historical and prediction data
    # (Code remains the same)
    logger.info("STEP 5: Combining historical and prediction data...")
    for col in historical_df.columns:
        if col not in prediction_df.columns: prediction_df[col] = np.nan
    common_cols = list(set(historical_df.columns).intersection(prediction_df.columns))
    if not common_cols: logger.error("No common columns found."); return False
    if 'is_home' in historical_df.columns and 'is_home' not in common_cols: common_cols.append('is_home')
    combined_df = pd.concat([historical_df[common_cols], prediction_df[common_cols]], ignore_index=True)
    combined_df['game_date'] = pd.to_datetime(combined_df['game_date']).dt.normalize()
    combined_df = combined_df.sort_values(by=['pitcher_id', 'game_date']).reset_index(drop=True)
    del historical_df, prediction_df; gc.collect()


    logger.info(f"DEBUG: Columns BEFORE Step 6 (Feature Engineering): {combined_df.columns.tolist()}") # ADD THIS

    # STEP 6: Engineer features on combined data
    logger.info("STEP 6: Engineering prediction features...")
    featured_df = combined_df.copy() # Use the combined data
    try:
        # Import feature creation functions (ensure they are imported)
        from src.features.pitcher_features import create_pitcher_features # Assuming this is the main wrapper now

        # Make sure featured_df is passed correctly
        # Ensure create_pitcher_features is designed to handle the combined df
        # and the necessary supplementary dataframes (team_game_stats, umpire_data)
        # Note: Minimal pitch data is NOT loaded here for prediction runs
        featured_df = create_pitcher_features(
            featured_df, # Pass the combined df
            pitch_level_data=None, # Explicitly None for prediction
            team_batting_data=team_game_stats, # Pass loaded team game stats
            umpire_data=umpire_data # Pass loaded umpire data
        )

    except Exception as e:
        logger.error(f"Error during feature creation step: {e}", exc_info=True)
        return False

    logger.info(f"DEBUG: Columns AFTER Step 6 (Feature Engineering): {featured_df.columns.tolist()}") # ADD THIS
    del combined_df; gc.collect()


    # STEP 7: Extract only prediction records for the target date
    # (Code remains the same)
    logger.info("STEP 7: Extracting prediction records...")
    prediction_features = featured_df[featured_df['game_date'] == prediction_dt].copy()
    del featured_df; gc.collect()
    if prediction_features.empty: logger.error(f"No prediction features generated for {prediction_date_str}."); return False
    logger.info(f"Generated {len(prediction_features)} records with features for prediction.")


    # STEP 8: Apply MULTIPLE target encodings using saved map
    # (Code remains the same - already handles multiple encodings)
    logger.info("STEP 8: Applying target encodings...")
    encoding_map = None
    try:
        models_dir = project_root / 'models'
        encoding_files = list(models_dir.glob('encoding_map_*.pkl'))
        if encoding_files:
            latest_encoding = max(encoding_files, key=lambda x: x.stat().st_mtime)
            with open(latest_encoding, 'rb') as f: encoding_map = pickle.load(f)
            logger.info(f"Loaded encoding map from {latest_encoding}")
        else: logger.warning("No encoding map found. Cannot apply target encodings.")
    except Exception as e: logger.error(f"Failed to load encoding map: {e}", exc_info=True)

    if encoding_map:
        for column, enc_data in encoding_map.items():
            logger.info(f"  Applying target encoding for column: {column}")
            if column in prediction_features.columns:
                col_map = enc_data.get('map', {}); default_val = enc_data.get('default', 0)
                if prediction_features[column].dtype == 'object': prediction_features[column] = prediction_features[column].astype(str)
                prediction_features[f'{column}_encoded'] = prediction_features[column].map(col_map).fillna(default_val)
                logger.info(f"    Applied encoding for {column}.")
            else: logger.warning(f"    Column '{column}' not found in prediction features. Skipping encoding for it.")


    # STEP 9: Handle final missing values
    # (Code remains the same)
    logger.info("STEP 9: Handling final missing values...")
    numeric_cols = prediction_features.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if prediction_features[col].isnull().sum() > 0:
            fill_value = 0
            logger.info(f"Filling {prediction_features[col].isnull().sum()} missing values in {col} with {fill_value}")
            prediction_features[col] = prediction_features[col].fillna(fill_value)


    # STEP 10: Saving prediction features
    logger.info("STEP 10: Saving prediction features...")

    # Ensure essential identifier columns are present (Keep this logic)
    essential_cols = ['gamePk', 'pitcher_id', 'game_date', 'is_home', 'player_name', 'home_team', 'away_team', 'season', 'opponent_team']
    existing_cols_in_df = [col for col in prediction_features.columns]
    cols_to_save = [col for col in essential_cols if col in existing_cols_in_df] + \
                   [col for col in existing_cols_in_df if col not in essential_cols]
    missing_essentials = [col for col in essential_cols if col not in prediction_features.columns]
    if missing_essentials:
        logger.warning(f"Essential columns missing before final selection: {missing_essentials}.")

    final_prediction_df_to_save = prediction_features[cols_to_save].copy()

    # Save Checkpoint (using the final df)
    try:
        logger.info(f"Saving prediction checkpoint: {prediction_checkpoint}")
        with open(prediction_checkpoint, 'wb') as f: pickle.dump(final_prediction_df_to_save, f)
        logger.info("Prediction checkpoint saved")
    except Exception as e: logger.error(f"Failed to save prediction checkpoint: {e}")

    # Save to Database (using the final df)
    try:
        with DBConnection() as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            logger.info("Saving prediction features to 'prediction_features' table...")

            # --- >>>> ADD THE DEBUG LINE HERE <<<< ---
            logger.info(f"DEBUG: Columns ACTUALLY being saved to DB: {final_prediction_df_to_save.columns.tolist()}")
            # --- >>>> END DEBUG LINE <<<< ---

            # Save the potentially modified dataframe
            final_prediction_df_to_save.to_sql('prediction_features', conn, if_exists='replace', index=False, chunksize=10000) # The line AFTER the debug print
            logger.info(f"Saved {len(final_prediction_df_to_save)} prediction records to database")
    except Exception as e: logger.error(f"Failed to save prediction features to database: {e}", exc_info=True); return False



    total_time = time.time() - start_time
    logger.info(f"=== Prediction Feature Generation Completed in {total_time:.2f}s ===")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLB feature engineering pipeline (SQL Aggregation).")
    parser.add_argument("--real-world", action="store_true", help="Generate features for prediction on a specific date.")
    parser.add_argument("--prediction-date", type=str, help="Date (YYYY-MM-DD) for prediction features.")
    parser.add_argument("--ignore-checkpoint", action="store_true", help="Ignore existing checkpoints and rebuild all features/aggregations.")

    args = parser.parse_args()
    success = False

    try:
        if args.real_world:
            if not args.prediction_date: logger.error("--prediction-date is required with --real-world"); sys.exit(1)
            success = generate_prediction_features(args.prediction_date, args) # <-- Pass args here
        else:
            # Assuming run_historical_feature_pipeline also needs args for its checkpoint logic
            success = run_historical_feature_pipeline(args)
    except Exception as e:
        logger.error(f"Unhandled exception in main execution: {e}", exc_info=True)
        success = False

    if success:
        logger.info("=== Feature Engineering Finished Successfully ===")
        sys.exit(0)
    else:
        logger.error("=== Feature Engineering Finished With Errors ===")
        sys.exit(1)
