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
    """Loads features for the specified prediction date."""
    logger.info(f"Loading prediction features for date: {prediction_date_str}")
    table_name = "prediction_features"
    try:
        with DBConnection(db_path) as conn:
             if conn is None: raise ConnectionError("DB Connection failed.")
             # Check if table exists first
             cursor = conn.cursor()
             cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
             if not cursor.fetchone():
                  logger.error(f"Prediction features table '{table_name}' not found.")
                  logger.error(f"Run 'engineer_features.py --real-world --prediction-date {prediction_date_str}' first.")
                  return None
             # Load features only for the specific date (though table should only contain that)
             query = f"SELECT * FROM {table_name} WHERE game_date = '{prediction_date_str}'"
             features_df = pd.read_sql_query(query, conn)
             if features_df.empty:
                  logger.warning(f"No prediction features found in '{table_name}' for {prediction_date_str}.")
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
    if not MODULE_IMPORTS_OK: logger.error("Exiting: Failed module imports."); return

    db_path = project_root / DBConfig.PATH
    prediction_date = args.prediction_date

    # 1. Load Prediction Features
    features_df = load_prediction_features(prediction_date, db_path)
    if features_df is None or features_df.empty:
        logger.error("Failed to load prediction features. Exiting.")
        return

    # 2. Load Model
    try:
        model_path = Path(args.model_path)
        if not model_path.exists(): raise FileNotFoundError()
        bst = lgb.Booster(model_file=str(model_path))
        logger.info(f"Loaded model from: {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found: {args.model_path}")
        return
    except Exception as e:
        logger.error(f"Error loading LightGBM model: {e}", exc_info=True)
        return

    # 3. Load Feature List used for Training
    try:
        features_list_path = Path(args.features_path)
        if not features_list_path.exists(): raise FileNotFoundError()
        with open(features_list_path, 'rb') as f:
            training_features = pickle.load(f)
        logger.info(f"Loaded {len(training_features)} feature names from: {features_list_path}")
    except FileNotFoundError:
        logger.error(f"Feature list file not found: {args.features_path}")
        return
    except Exception as e:
        logger.error(f"Error loading feature list: {e}", exc_info=True)
        return

    # 4. Prepare Features for Prediction
    # Ensure all required training features are present, fill missing with 0 (or median if stored)
    missing_cols = list(set(training_features) - set(features_df.columns))
    if missing_cols:
        logger.warning(f"Prediction features are missing {len(missing_cols)} columns expected by model: {missing_cols[:5]}...")
        for col in missing_cols:
            features_df[col] = 0 # Or load imputation values if saved during training
    # Select only the features the model expects, in the correct order
    try:
        predict_X = features_df[training_features]
    except KeyError as e:
         logger.error(f"Column mismatch preparing features: {e}. Ensure prediction features table and loaded feature list match.")
         return

    # 5. Make Predictions
    logger.info("Generating predictions...")
    predictions = bst.predict(predict_X, num_iteration=bst.best_iteration) # Use best iteration if available
    features_df['predicted_strikeouts'] = predictions
    logger.info("Predictions generated.")

    # 6. Load Identifiers for Output
    id_df = load_game_identifiers(prediction_date, db_path)

    # 7. Combine Predictions with Identifiers
    output_df = pd.DataFrame()
    if id_df is not None and not id_df.empty:
         # Need to map predictions back to games/pitchers
         # The features_df has gamePk, pitcher_id, is_home
         output_df = pd.merge(
              features_df[['gamePk', 'pitcher_id', 'is_home', 'predicted_strikeouts']],
              id_df,
              on='gamePk',
              how='left'
         )
         # Separate home/away predictions and add pitcher names etc.
         home_preds = output_df[output_df['is_home'] == 1].copy()
         away_preds = output_df[output_df['is_home'] == 0].copy()

         # Rename columns for clarity before potential final merge (or keep separate)
         home_preds = home_preds[['gamePk', 'home_team_abbr', 'away_team_abbr', 'home_probable_pitcher_id', 'home_probable_pitcher_name', 'predicted_strikeouts']]
         home_preds.rename(columns={'home_team_abbr':'Team', 'away_team_abbr':'Opponent', 'home_probable_pitcher_id':'PitcherID', 'home_probable_pitcher_name':'PitcherName', 'predicted_strikeouts':'Predicted_SO'}, inplace=True)

         away_preds = away_preds[['gamePk', 'home_team_abbr', 'away_team_abbr', 'away_probable_pitcher_id', 'away_probable_pitcher_name', 'predicted_strikeouts']]
         away_preds.rename(columns={'away_team_abbr':'Team', 'home_team_abbr':'Opponent', 'away_probable_pitcher_id':'PitcherID', 'away_probable_pitcher_name':'PitcherName', 'predicted_strikeouts':'Predicted_SO'}, inplace=True)

         final_output_df = pd.concat([home_preds, away_preds], ignore_index=True).sort_values(by='gamePk').reset_index(drop=True)
         # Optional: Round predictions
         final_output_df['Predicted_SO'] = final_output_df['Predicted_SO'].round(2)

    else:
         logger.warning("Could not load game identifiers. Outputting predictions with limited info.")
         # Fallback: output only pitcher IDs and predictions
         final_output_df = features_df[['pitcher_id', 'gamePk', 'is_home', 'predicted_strikeouts']].copy()
         final_output_df.rename(columns={'predicted_strikeouts':'Predicted_SO'}, inplace=True)


    # 8. Save or Print Output
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
        print(final_output_df.to_string(index=False))


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