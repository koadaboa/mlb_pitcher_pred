# src/scripts/predict_today.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import logging
import argparse
from pathlib import Path
import sys
from datetime import datetime

# Ensure src directory is in the path if running script directly
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Attempt imports
try:
    from src.data.utils import setup_logger, DBConnection
    from src.config import DBConfig
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    MODULE_IMPORTS_OK = False
    def setup_logger(name, level=logging.INFO, log_file=None): logging.basicConfig(level=level); return logging.getLogger(name)
    class DBConnection: # Dummy
        def __init__(self, p): self.p=p; print(f"WARN: Using dummy DBConnection for {p}")
        def __enter__(self): return None
        def __exit__(self,t,v,tb): pass
    class DBConfig: PATH="data/pitcher_stats.db"

# Setup logger
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True, parents=True)
logger = setup_logger('predict_today', log_file= log_dir / 'predict_today.log', level=logging.INFO) if MODULE_IMPORTS_OK else logging.getLogger('predict_fallback')


def load_prediction_features(prediction_date_str, db_path):
    """Loads features for the specified prediction date using parameterized query."""
    logger.info(f"Loading prediction features for date: {prediction_date_str}")
    table_name = "prediction_features"
    try:
        with DBConnection(db_path) as conn:
             if conn is None: raise ConnectionError("DB Connection failed.")
             cursor = conn.cursor()
             cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
             if not cursor.fetchone():
                  logger.error(f"Prediction features table '{table_name}' not found.")
                  logger.error(f"Run 'engineer_features.py --real-world --prediction-date {prediction_date_str}' first.")
                  return None

             query = f"SELECT * FROM {table_name} WHERE date(game_date) = ?"
             features_df = pd.read_sql_query(query, conn, params=(prediction_date_str,))

             # --- >>>> ADD THIS DEBUG LINE <<<< ---
             logger.info(f"DEBUG: Columns LOADED into features_df: {features_df.columns.tolist()}")
             # --- >>>> END DEBUG LINE <<<< ---

             if features_df.empty:
                  logger.warning(f"No prediction features found in '{table_name}' for {prediction_date_str} using query: {query} with param: {prediction_date_str}")
                  return None
             logger.info(f"Loaded {len(features_df)} feature rows for prediction.")
             return features_df
    except Exception as e:
        logger.error(f"Error loading prediction features from {table_name}: {e}", exc_info=True)
        return None

def load_game_identifiers(prediction_date_str, db_path):
     """Loads game identifiers (teams, pitchers) from mlb_api table for the prediction date."""
     logger.info(f"Loading game/pitcher identifiers for date: {prediction_date_str}")
     table_name = "mlb_api"
     try:
        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone():
                logger.error(f"Identifiers table '{table_name}' not found.")
                logger.error(f"Run 'data_fetcher.py --mlb-api --date {prediction_date_str}' first.")
                return None
            query = f"SELECT gamePk, home_team_abbr, away_team_abbr, home_probable_pitcher_id, home_probable_pitcher_name, away_probable_pitcher_id, away_probable_pitcher_name FROM {table_name} WHERE game_date = '{prediction_date_str}'"
            id_df = pd.read_sql_query(query, conn)
            if id_df.empty:
                 logger.warning(f"No game identifiers found in '{table_name}' for {prediction_date_str}.")
                 return None
            logger.info(f"Loaded {len(id_df)} games' identifiers.")
            # Ensure IDs are nullable integers for merging
            for col in ['home_probable_pitcher_id', 'away_probable_pitcher_id']:
                 id_df[col] = pd.to_numeric(id_df[col], errors='coerce').astype('Int64')
            return id_df
     except Exception as e:
        logger.error(f"Error loading game identifiers from {table_name}: {e}", exc_info=True)
        return None


def generate_predictions(args):
    """Loads model, features, makes predictions, and saves/prints."""
    # Ensure necessary modules/functions are available if run standalone
    if 'logger' not in globals(): # Basic check
        print("Error: logger not defined.")
        return
    if 'DBConfig' not in globals():
         print("Error: DBConfig not defined.")
         return
    # Add similar checks for DBConnection, load_prediction_features, load_game_identifiers if needed

    db_path = Path(DBConfig.PATH) # Use Path object
    prediction_date = args.prediction_date

    # 1. Load Prediction Features
    features_df = load_prediction_features(prediction_date, db_path)
    if features_df is None or features_df.empty:
        logger.error("Failed to load prediction features. Exiting.")
        return

    # 2. Load Model
    try:
        model_path = Path(args.model_path)
        if not model_path.exists(): raise FileNotFoundError(f"Model file not found: {model_path}")
        bst = lgb.Booster(model_file=str(model_path))
        logger.info(f"Loaded model from: {model_path}")
    except FileNotFoundError as e:
        logger.error(e)
        return
    except Exception as e:
        logger.error(f"Error loading LightGBM model: {e}", exc_info=True)
        return

    # 3. Load Feature List used for Training
    try:
        features_list_path = Path(args.features_path)
        if not features_list_path.exists(): raise FileNotFoundError(f"Feature list file not found: {features_list_path}")
        with open(features_list_path, 'rb') as f:
            training_features = pickle.load(f)
        logger.info(f"Loaded {len(training_features)} feature names from: {features_list_path}")
    except FileNotFoundError as e:
        logger.error(e)
        return
    except Exception as e:
        logger.error(f"Error loading feature list: {e}", exc_info=True)
        return

    # 4. Prepare Features for Prediction
    # Ensure all required training features are present
    logger.info(f"DEBUG: Columns LOADED into features_df: {features_df.columns.tolist()}") # Keep this debug line
    loaded_feature_cols = list(features_df.columns)
    missing_model_cols = list(set(training_features) - set(loaded_feature_cols))

    if missing_model_cols:
        logger.warning(f"Prediction features are missing {len(missing_model_cols)} columns expected by model: {missing_model_cols[:5]}...")
        for col in missing_model_cols:
            logger.info(f"Adding missing column '{col}' with default value 0.")
            features_df[col] = 0 # Add missing column, fill with 0

    # Ensure columns are in the same order as training
    try:
        # Select only the features the model expects, in the correct order
        predict_X = features_df[training_features]
    except KeyError as e:
         logger.error(f"Column mismatch preparing features even after adding missing: {e}. Check feature list consistency.", exc_info=True)
         logger.error(f"Columns available in features_df: {features_df.columns.tolist()}")
         logger.error(f"Columns expected by model (training_features): {training_features}")
         return

    # 5. Make Predictions
    logger.info("Generating predictions...")
    try:
        predictions = bst.predict(predict_X, num_iteration=bst.best_iteration) # Use best iteration if available
        features_df['predicted_strikeouts'] = predictions
        logger.info("Predictions generated.")
    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        return


    # 6. Load Identifiers for Output
    id_df = load_game_identifiers(prediction_date, db_path)

    # 7. Combine Predictions with Identifiers
    final_output_df = pd.DataFrame() # Initialize final output DataFrame

    if id_df is not None and not id_df.empty:
         logger.info("Merging predictions with game identifiers...")
         # --- Define columns needed from features_df for the merge ---
         cols_for_merge = ['pitcher_id', 'is_home', 'predicted_strikeouts']
         merge_on_key = None # Initialize merge key

         # --- Use lowercase 'game_pk' consistently ---
         if 'game_pk' in features_df.columns:
              cols_for_merge.insert(0, 'game_pk') # Add lowercase 'game_pk'
              merge_on_key = 'game_pk'            # Use lowercase 'game_pk' as the key
              logger.debug(f"Found 'game_pk' in features_df. Using it as merge key.")
         else:
              # This should ideally not be reached if engineer_features is fixed
              logger.error("FATAL: 'game_pk' column (lowercase) still missing from loaded features_df. Cannot reliably merge results.")
              # Fallback: Output predictions without full game context
              final_output_df = features_df[['pitcher_id', 'is_home', 'predicted_strikeouts']].copy()
              final_output_df.rename(columns={'predicted_strikeouts':'Predicted_SO'}, inplace=True)
              logger.warning("Outputting predictions with limited info due to missing merge key.")
              # Go directly to saving/printing this limited output
              # (Skip the rest of the merge/rename logic within this 'if id_df' block)

         # --- Proceed with merge only if merge_on_key was set ---
         if merge_on_key:
             logger.debug(f"Columns selected for merge from features_df: {cols_for_merge}")
             logger.debug(f"Merge key: {merge_on_key}")

             try:
                 # --- Ensure id_df uses the same merge key case ('game_pk') ---
                 # Check if id_df has 'gamePk' (uppercase P) and rename it if needed
                 if 'gamePk' in id_df.columns and 'game_pk' not in id_df.columns:
                      logger.debug("Renaming 'gamePk' to 'game_pk' in id_df for merging.")
                      id_df = id_df.rename(columns={'gamePk': 'game_pk'})
                 elif 'gamePk' in id_df.columns and 'game_pk' in id_df.columns and 'gamePk' != 'game_pk':
                      logger.warning("Both 'gamePk' and 'game_pk' found in id_df. Using 'game_pk'.")
                      id_df = id_df.drop(columns=['gamePk']) # Drop the potential duplicate uppercase version
                 elif 'game_pk' not in id_df.columns:
                      logger.error(f"Merge key '{merge_on_key}' not found in id_df columns: {id_df.columns.tolist()}")
                      raise ValueError(f"Merge key '{merge_on_key}' missing in identifier data.")

                 # --- Perform the merge ---
                 subset_df_for_merge = features_df[cols_for_merge].copy() # Create subset safely
                 output_df = pd.merge(
                      subset_df_for_merge,
                      id_df,
                      on=merge_on_key, # Merge on lowercase 'game_pk'
                      how='left'
                 )
                 logger.info(f"Merge successful. Shape after merge: {output_df.shape}")

                 # --- Process merged results ---
                 home_preds = output_df[output_df['is_home'] == 1].copy()
                 away_preds = output_df[output_df['is_home'] == 0].copy()

                 # Define expected columns from id_df after potential rename
                 id_game_pk_col = 'game_pk' # Use the consistent lowercase key

                 # Prepare home results
                 home_cols_needed = [id_game_pk_col, 'home_team_abbr', 'away_team_abbr', 'home_probable_pitcher_id', 'home_probable_pitcher_name', 'predicted_strikeouts']
                 missing_home_cols = [c for c in home_cols_needed if c not in home_preds.columns]
                 if missing_home_cols: logger.warning(f"Missing columns in home_preds for final output: {missing_home_cols}")

                 home_preds_renamed = home_preds[[c for c in home_cols_needed if c in home_preds.columns]].rename(
                     columns={id_game_pk_col: 'gamePk', # Rename back to 'gamePk' for output if desired
                              'home_team_abbr':'Team',
                              'away_team_abbr':'Opponent',
                              'home_probable_pitcher_id':'PitcherID',
                              'home_probable_pitcher_name':'PitcherName',
                              'predicted_strikeouts':'Predicted_SO'})

                 # Prepare away results
                 away_cols_needed = [id_game_pk_col, 'home_team_abbr', 'away_team_abbr', 'away_probable_pitcher_id', 'away_probable_pitcher_name', 'predicted_strikeouts']
                 missing_away_cols = [c for c in away_cols_needed if c not in away_preds.columns]
                 if missing_away_cols: logger.warning(f"Missing columns in away_preds for final output: {missing_away_cols}")

                 away_preds_renamed = away_preds[[c for c in away_cols_needed if c in away_preds.columns]].rename(
                     columns={id_game_pk_col: 'gamePk', # Rename back to 'gamePk' for output if desired
                              'away_team_abbr':'Team',
                              'home_team_abbr':'Opponent',
                              'away_probable_pitcher_id':'PitcherID',
                              'away_probable_pitcher_name':'PitcherName',
                              'predicted_strikeouts':'Predicted_SO'})

                 # Combine and finalize
                 final_output_df = pd.concat([home_preds_renamed, away_preds_renamed], ignore_index=True)
                 # Ensure gamePk is integer type if present
                 if 'gamePk' in final_output_df.columns:
                      final_output_df['gamePk'] = pd.to_numeric(final_output_df['gamePk'], errors='coerce').astype('Int64')
                 final_output_df = final_output_df.sort_values(by='gamePk' if 'gamePk' in final_output_df.columns else 'PitcherID').reset_index(drop=True)

                 # Optional: Round predictions
                 if 'Predicted_SO' in final_output_df.columns:
                     final_output_df['Predicted_SO'] = final_output_df['Predicted_SO'].round(2)

             except Exception as merge_err:
                 logger.error(f"Error during merge/rename operation: {merge_err}", exc_info=True)
                 # Fallback if merge still fails
                 final_output_df = features_df[['pitcher_id', 'is_home', 'predicted_strikeouts']].copy()
                 final_output_df.rename(columns={'predicted_strikeouts':'Predicted_SO'}, inplace=True)
                 logger.warning("Outputting predictions with limited info due to merge error.")

    # --- Handle case where id_df failed to load initially ---
    else:
         logger.warning("Could not load game identifiers. Outputting predictions with limited info.")
         final_output_df = features_df[['pitcher_id', 'is_home', 'predicted_strikeouts']].copy()
         final_output_df.rename(columns={'predicted_strikeouts':'Predicted_SO'}, inplace=True)


    # 8. Save or Print Output
    if not final_output_df.empty:
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = args.output_format.lower()
            try:
                if fmt == 'csv':
                    final_output_df.to_csv(output_path, index=False)
                    logger.info(f"Predictions saved to CSV: {output_path}")
                elif fmt == 'json':
                    final_output_df.to_json(output_path, orient='records', indent=2)
                    logger.info(f"Predictions saved to JSON: {output_path}")
                else:
                     logger.error(f"Unsupported output format: {args.output_format}")
            except Exception as e:
                 logger.error(f"Failed to save predictions to {output_path}: {e}")
        else:
            # Print to console if no output file specified
            print("\n--- Strikeout Predictions ---")
            # Select and order columns for printing
            cols_to_print = ['gamePk', 'Team', 'Opponent', 'PitcherID', 'PitcherName', 'Predicted_SO']
            # Filter to only columns that actually exist in the final df
            printable_cols = [col for col in cols_to_print if col in final_output_df.columns]
            print(final_output_df[printable_cols].to_string(index=False))

        logger.info("Attempting to save predictions to database...")
        today_str = datetime.now().strftime('%Y-%m-%d') # Get current date for run date
        save_success = save_predictions_to_db(
            predictions_df=final_output_df,
            prediction_run_date_str=today_str,
            game_date_str=args.prediction_date, # The date the predictions are for
            model_path_str=args.model_path,     # Pass model path for versioning
            db_path=db_path
        )
        if not save_success:
             logger.error("Failed to save predictions to database.")
    else:
         logger.error("Final output DataFrame is empty. No predictions to show/save.")

def save_predictions_to_db(predictions_df, prediction_run_date_str, game_date_str, model_path_str, db_path):
    """
    Formats predictions and saves them to the daily_predictions table in the DB.

    Args:
        predictions_df (pd.DataFrame): DataFrame containing the final predictions
                                      (columns like gamePk, Team, Opponent, PitcherID,
                                       PitcherName, Predicted_SO).
        prediction_run_date_str (str): The date the script is being run (YYYY-MM-DD).
        game_date_str (str): The date the predictions are for (YYYY-MM-DD).
        model_path_str (str): Path to the model file used for prediction.
        db_path (str or Path): Path to the SQLite database file.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if predictions_df.empty:
        logger.warning("No predictions to save to database.")
        return False

    table_name = "daily_predictions"
    logger.info(f"Preparing to save {len(predictions_df)} predictions to table '{table_name}'...")

    try:
        # Create a copy to avoid modifying the original DataFrame
        df_to_save = predictions_df.copy()

        # Add necessary metadata columns
        df_to_save['prediction_run_date'] = prediction_run_date_str
        df_to_save['game_date'] = game_date_str
        # Extract model identifier (e.g., timestamp) from path
        model_version = Path(model_path_str).stem # Gets filename without extension
        df_to_save['model_version'] = model_version
        df_to_save['actual_strikeouts'] = None # Placeholder for actuals

        # Rename columns to match target table schema
        df_to_save = df_to_save.rename(columns={
            'gamePk': 'gamePk', # Keep consistent or change if needed
            'Team': 'team_abbr',
            'Opponent': 'opponent_abbr',
            'PitcherID': 'pitcher_id',
            'PitcherName': 'pitcher_name',
            'Predicted_SO': 'predicted_strikeouts'
        })

        # Select and order columns for the database table
        final_cols = [
            'prediction_run_date', 'game_date', 'gamePk', 'pitcher_id',
            'pitcher_name', 'team_abbr', 'opponent_abbr',
            'predicted_strikeouts', 'model_version', 'actual_strikeouts'
        ]
        # Ensure only existing columns are selected
        cols_to_save_final = [col for col in final_cols if col in df_to_save.columns]
        missing_final_cols = [col for col in final_cols if col not in cols_to_save_final]
        if missing_final_cols:
             logger.warning(f"Could not save columns to DB, missing from source df: {missing_final_cols}")

        df_to_save = df_to_save[cols_to_save_final]

        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            cursor = conn.cursor()

            # Create table if it doesn't exist
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                prediction_run_date TEXT,
                game_date TEXT,
                gamePk INTEGER,
                pitcher_id INTEGER,
                pitcher_name TEXT,
                team_abbr TEXT,
                opponent_abbr TEXT,
                predicted_strikeouts REAL,
                model_version TEXT,
                actual_strikeouts INTEGER,
                PRIMARY KEY (game_date, gamePk, pitcher_id, prediction_run_date, model_version)
            );
            """)
            conn.commit()

            # Append data - consider adding conflict handling if needed
            # e.g., INSERT OR IGNORE or INSERT OR REPLACE
            logger.info(f"Appending {len(df_to_save)} records to '{table_name}'...")
            df_to_save.to_sql(table_name, conn, if_exists='append', index=False)
            logger.info(f"Successfully saved predictions to '{table_name}'.")
            return True

    except Exception as e:
        logger.error(f"Error saving predictions to database table '{table_name}': {e}", exc_info=True)
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MLB Strikeout Predictions for a specific date.")
    parser.add_argument(
        "--prediction-date",
        type=str,
        required=True,
        help="Date (YYYY-MM-DD) to generate predictions for."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained LightGBM model file (.txt)."
    )
    parser.add_argument(
        "--features-path",
        type=str,
        required=True,
        help="Path to the list of feature columns used during training (.pkl)."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Optional path to save predictions (e.g., predictions/preds_YYYY-MM-DD.csv). If not provided, prints to console."
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="csv",
        choices=['csv', 'json'],
        help="Format for the output file (csv or json). Default: csv."
    )
    return parser.parse_args()

if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")

    args = parse_args()
    # Validate date format
    try: datetime.strptime(args.prediction_date, "%Y-%m-%d")
    except ValueError: logger.error(f"Invalid format for --prediction-date: {args.prediction_date}. Use YYYY-MM-DD."); sys.exit(1)

    generate_predictions(args)
    logger.info("Prediction script finished.")