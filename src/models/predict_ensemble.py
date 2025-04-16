# src/scripts/predict_ensemble.py

# Import necessary libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pickle
import logging
import argparse
from pathlib import Path
import sys
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error # For evaluation
import re

# --- Assume src is in Python path ---
try:
    from src.config import DBConfig, StrikeoutModelConfig # Need config for test years
    from src.data.utils import setup_logger, DBConnection
    # Import loading function for identifiers (still needed for context if saving eval results)
    from src.scripts.predict_today import load_game_identifiers
except ImportError:
    print("Error: Could not import dependencies from src.", file=sys.stderr)
    # Define fallbacks
    class DBConfig: PATH = "data/pitcher_stats.db"
    class StrikeoutModelConfig: DEFAULT_TEST_YEARS = (2024, 2025) # Example fallback
    class DBConnection:
        def __init__(self, p): self.p=p
        def __enter__(self): self.conn = sqlite3.connect(self.p); return self.conn
        def __exit__(self,t,v,tb):
             if self.conn: self.conn.close()
    def setup_logger(n,log_file=None, level=logging.INFO): logging.basicConfig(level=level); return logging.getLogger(n)
    def load_game_identifiers(d,p): return pd.DataFrame() # Dummy
    # Need mean_squared_error, mean_absolute_error from sklearn.metrics

# Setup logger
log_dir = Path(DBConfig.PATH).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True, parents=True)
logger = setup_logger('predict_ensemble', log_file= log_dir / 'predict_ensemble.log', level=logging.INFO)


# --- Helper Function (Optional but good for evaluation) ---
def within_n_strikeouts(y_true, y_pred, n=1):
    """ Calculates percentage of predictions within n strikeouts """
    if y_true is None or y_pred is None or len(y_true) != len(y_pred):
        return np.nan
    y_true_arr = np.asarray(y_true); y_pred_arr = np.asarray(y_pred)
    within_n = np.abs(y_true_arr - np.round(y_pred_arr)) <= n
    return np.mean(within_n)

def find_latest_file(directory, pattern):
    model_dir = Path(directory); files = list(model_dir.glob(pattern))
    if not files: return None
    ts_pattern = re.compile(r"_(\d{8}_\d{6})\."); latest_file, latest_timestamp = None, 0; parsed = False
    for f in files:
        match = ts_pattern.search(f.name)
        if match:
            try: ts = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").timestamp(); latest_file = f if ts > latest_timestamp else latest_file; latest_timestamp = max(ts, latest_timestamp); parsed = True
            except ValueError: pass
    if not parsed and files: latest_file = max(files, key=lambda x: x.stat().st_mtime)
    return latest_file
# --- End Helper ---


def generate_ensemble_predictions_or_evaluate(args):
    """Loads features & models. Handles separate feature lists."""
    db_path = Path(DBConfig.PATH)
    features_df = pd.DataFrame()
    y_true = None
    model_dir = Path(DBConfig.PATH).parent.parent / 'models' # Define model dir

    # --- Mode Logic: Load appropriate data ---
    if args.evaluate_test_set:
        # ... (Load features_df and y_true from test_features table - NO CHANGE NEEDED HERE) ...
        mode = "Evaluation"
        logger.info("--- Running in EVALUATION MODE on Test Set ---")
        table_name = "test_features"
        try:
            with DBConnection(db_path) as conn:
                 if conn is None: raise ConnectionError("DB Connection failed.")
                 cursor = conn.cursor(); cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                 if not cursor.fetchone(): raise FileNotFoundError(f"Table '{table_name}' not found.")
                 logger.info(f"Loading features and actuals from: {table_name}")
                 features_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                 if 'strikeouts' not in features_df.columns: raise ValueError(f"Target 'strikeouts' not found in {table_name}.")
                 y_true = features_df['strikeouts'].copy()
                 logger.info(f"Loaded {len(features_df)} rows for evaluation.")
        except Exception as e: logger.error(f"Error loading data from {table_name}: {e}", exc_info=True); return

    elif args.prediction_date:
         # ... (Load features_df from prediction_features table - NO CHANGE NEEDED HERE) ...
        mode = "Prediction"
        logger.info(f"--- Running in PREDICTION MODE for Date: {args.prediction_date} ---")
        # Reuse the loading function if preferred
        features_df = load_prediction_features(args.prediction_date, db_path) # Assumes this function exists
        if features_df is None or features_df.empty: logger.error("Failed to load prediction features."); return
    else:
        logger.error("Invalid mode."); return

    # --- Load Models (Automatically find latest) ---
    try:
        lgbm_pattern = "*_strikeout_model_*.txt"
        lgbm_model_path = find_latest_file(model_dir, lgbm_pattern)
        if not lgbm_model_path: raise FileNotFoundError(f"Could not find LightGBM model matching '{lgbm_pattern}'")
        lgbm_bst = lgb.Booster(model_file=str(lgbm_model_path))
        logger.info(f"Loaded LightGBM model from: {lgbm_model_path}")

        xgb_pattern = "xgboost_strikeout_model_*.json" # Or *.ubj
        xgb_model_path = find_latest_file(model_dir, xgb_pattern)
        if not xgb_model_path: raise FileNotFoundError(f"Could not find XGBoost model matching '{xgb_pattern}'")
        xgb_bst = xgb.Booster(); xgb_bst.load_model(str(xgb_model_path))
        logger.info(f"Loaded XGBoost model from: {xgb_model_path}")

    except FileNotFoundError as e: logger.error(e); return
    except Exception as e: logger.error(f"Error loading one or more models: {e}", exc_info=True); return

    # --- Load Feature Lists (Load one for EACH model) ---
    try:
        # Find feature list associated with the loaded LGBM model
        lgbm_features_pattern = f"{lgbm_model_path.stem.replace('strikeout_model','feature_columns')}.pkl"
        # Or use a more generic pattern if timestamps might not exactly match
        # lgbm_features_pattern = "test_feature_columns_*.pkl" # Or "lgbm_feature_columns_*.pkl"
        lgbm_features_list_path = find_latest_file(model_dir, lgbm_features_pattern) # Use pattern specific to LGBM training output
        if not lgbm_features_list_path: raise FileNotFoundError(f"Could not find feature list matching '{lgbm_features_pattern}'")
        with open(lgbm_features_list_path, 'rb') as f: lgbm_training_features = pickle.load(f)
        logger.info(f"Loaded {len(lgbm_training_features)} LGBM feature names from: {lgbm_features_list_path}")

        # Find feature list associated with the loaded XGBoost model
        xgb_features_pattern = f"{xgb_model_path.stem.replace('strikeout_model','feature_columns')}.pkl"
        # Or use a more generic pattern
        # xgb_features_pattern = "xgboost_feature_columns_*.pkl"
        xgb_features_list_path = find_latest_file(model_dir, xgb_features_pattern)
        if not xgb_features_list_path: raise FileNotFoundError(f"Could not find feature list matching '{xgb_features_pattern}'")
        with open(xgb_features_list_path, 'rb') as f: xgb_training_features = pickle.load(f)
        logger.info(f"Loaded {len(xgb_training_features)} XGB feature names from: {xgb_features_list_path}")

    except FileNotFoundError as e: logger.error(e); return
    except Exception as e: logger.error(f"Error loading feature lists: {e}", exc_info=True); return

    # --- Prepare Data and Predict (Separately for each model) ---
    logger.info("Generating predictions from individual models...")
    try:
        # Prepare data for LGBM
        missing_lgbm_cols = list(set(lgbm_training_features) - set(features_df.columns))
        if missing_lgbm_cols:
            logger.warning(f"Adding {len(missing_lgbm_cols)} missing columns for LGBM (e.g., {missing_lgbm_cols[:3]}) with default 0.")
            for col in missing_lgbm_cols: features_df[col] = 0
        predict_X_lgbm = features_df[lgbm_training_features] # Select & Order LGBM features
        lgbm_preds = lgbm_bst.predict(predict_X_lgbm, num_iteration=lgbm_bst.best_iteration)
        logger.info("Generated LightGBM predictions.")

        # Prepare data for XGBoost
        missing_xgb_cols = list(set(xgb_training_features) - set(features_df.columns))
        if missing_xgb_cols:
             logger.warning(f"Adding {len(missing_xgb_cols)} missing columns for XGBoost (e.g., {missing_xgb_cols[:3]}) with default 0.")
             for col in missing_xgb_cols: features_df[col] = 0 # Add any missing for XGB
        predict_X_xgb = features_df[xgb_training_features] # Select & Order XGB features
        dmatrix_predict = xgb.DMatrix(predict_X_xgb, feature_names=xgb_training_features) # Pass feature names
        xgb_preds = xgb_bst.predict(dmatrix_predict)
        logger.info("Generated XGBoost predictions.")

    except Exception as e:
        logger.error(f"Error during model prediction preparation or execution: {e}", exc_info=True); return

    # --- Ensemble and Output/Evaluate ---
    logger.info("Averaging predictions...")
    ensemble_preds = (lgbm_preds + xgb_preds) / 2.0
    # Store ensemble prediction in the original features_df for merging/output
    features_df['predicted_strikeouts_ensemble'] = ensemble_preds

    if mode == "Evaluation":
        # ... (Evaluation metric calculation and printing - NO CHANGE NEEDED HERE) ...
        logger.info("--- Evaluating Ensemble Model on Test Set ---")
        if y_true is None: logger.error("Cannot evaluate, y_true missing."); return
        test_rmse = np.sqrt(mean_squared_error(y_true, ensemble_preds)); test_mae = mean_absolute_error(y_true, ensemble_preds)
        test_w1 = within_n_strikeouts(y_true, ensemble_preds, n=1); test_w2 = within_n_strikeouts(y_true, ensemble_preds, n=2)
        logger.info(f"Ensemble Test RMSE : {test_rmse:.4f}"); logger.info(f"Ensemble Test MAE  : {test_mae:.4f}")
        logger.info(f"Ensemble Test W/1 K: {test_w1:.4f}"); logger.info(f"Ensemble Test W/2 K: {test_w2:.4f}")
        # Optionally save results
        if args.output_file:
             # ... (save logic - ensure it uses 'predicted_strikeouts_ensemble') ...
             eval_output_df = features_df.copy(); eval_output_df['actual_strikeouts'] = y_true
             eval_output_df = eval_output_df.rename(columns={'predicted_strikeouts_ensemble':'Predicted_SO_Ensemble'})
             cols_to_keep = ['game_pk', 'pitcher_id', 'game_date', 'player_name', 'home_team', 'away_team', 'is_home', 'actual_strikeouts', 'Predicted_SO_Ensemble']
             eval_output_df = eval_output_df[[c for c in cols_to_keep if c in eval_output_df.columns]]
             output_path = Path(args.output_file); output_path.parent.mkdir(parents=True, exist_ok=True); fmt = args.output_format.lower()
             try:
                  if fmt == 'csv': eval_output_df.to_csv(output_path, index=False)
                  elif fmt == 'json': eval_output_df.to_json(output_path, orient='records', indent=2)
                  logger.info(f"Evaluation results saved to {fmt.upper()}: {output_path}")
             except Exception as e: logger.error(f"Failed to save evaluation results: {e}")


    elif mode == "Prediction":
        # ... (Prediction merging and output logic - NO CHANGE NEEDED, but uses 'predicted_strikeouts_ensemble') ...
        logger.info("--- Finalizing Predictions for Output ---")
        id_df = load_game_identifiers(args.prediction_date, db_path) # Assumes this function exists
        final_output_df = pd.DataFrame()
        if id_df is not None and not id_df.empty:
            cols_for_merge = ['pitcher_id', 'is_home', 'predicted_strikeouts_ensemble'] # Use ensemble pred col
            merge_on_key = None
            if 'game_pk' in features_df.columns: cols_for_merge.insert(0, 'game_pk'); merge_on_key = 'game_pk'
            else: logger.error("FATAL: 'game_pk' missing. Cannot merge.")

            if merge_on_key:
                try:
                    if 'gamePk' in id_df.columns and 'game_pk' not in id_df.columns: id_df = id_df.rename(columns={'gamePk': 'game_pk'})
                    elif 'game_pk' not in id_df.columns: raise ValueError("Merge key missing in identifier data.")
                    subset_df = features_df[cols_for_merge].copy()
                    output_df = pd.merge(subset_df, id_df, on=merge_on_key, how='left')
                    home_preds = output_df[output_df['is_home'] == 1].copy(); away_preds = output_df[output_df['is_home'] == 0].copy()
                    id_game_pk_col = 'game_pk'
                    home_cols_needed = [id_game_pk_col, 'home_team_abbr', 'away_team_abbr', 'home_probable_pitcher_id', 'home_probable_pitcher_name', 'predicted_strikeouts_ensemble']
                    away_cols_needed = [id_game_pk_col, 'home_team_abbr', 'away_team_abbr', 'away_probable_pitcher_id', 'away_probable_pitcher_name', 'predicted_strikeouts_ensemble']
                    home_preds_renamed = home_preds[[c for c in home_cols_needed if c in home_preds.columns]].rename(columns={id_game_pk_col: 'gamePk','home_team_abbr':'Team','away_team_abbr':'Opponent','home_probable_pitcher_id':'PitcherID','home_probable_pitcher_name':'PitcherName','predicted_strikeouts_ensemble':'Predicted_SO_Ensemble'})
                    away_preds_renamed = away_preds[[c for c in away_cols_needed if c in away_preds.columns]].rename(columns={id_game_pk_col: 'gamePk','away_team_abbr':'Team','home_team_abbr':'Opponent','away_probable_pitcher_id':'PitcherID','away_probable_pitcher_name':'PitcherName','predicted_strikeouts_ensemble':'Predicted_SO_Ensemble'})
                    final_output_df = pd.concat([home_preds_renamed, away_preds_renamed], ignore_index=True)
                    if 'gamePk' in final_output_df.columns: final_output_df['gamePk'] = pd.to_numeric(final_output_df['gamePk'], errors='coerce').astype('Int64')
                    final_output_df = final_output_df.sort_values(by='gamePk' if 'gamePk' in final_output_df.columns else 'PitcherID').reset_index(drop=True)
                    if 'Predicted_SO_Ensemble' in final_output_df.columns: final_output_df['Predicted_SO_Ensemble'] = final_output_df['Predicted_SO_Ensemble'].round(2)
                except Exception as merge_err:
                    logger.error(f"Error during merge/rename: {merge_err}", exc_info=True)
                    final_output_df = features_df[['pitcher_id', 'is_home', 'predicted_strikeouts_ensemble']].copy()
                    final_output_df.rename(columns={'predicted_strikeouts_ensemble':'Predicted_SO_Ensemble'}, inplace=True)
        else:
             logger.warning("Could not load game identifiers. Outputting predictions with limited info.")
             final_output_df = features_df[['pitcher_id', 'is_home', 'predicted_strikeouts_ensemble']].copy()
             final_output_df.rename(columns={'predicted_strikeouts_ensemble':'Predicted_SO_Ensemble'}, inplace=True)

        # Save or Print Prediction Output
        if not final_output_df.empty:
            if args.output_file:
                 output_path = Path(args.output_file); output_path.parent.mkdir(parents=True, exist_ok=True); fmt = args.output_format.lower()
                 try:
                     if fmt == 'csv': final_output_df.to_csv(output_path, index=False)
                     elif fmt == 'json': final_output_df.to_json(output_path, orient='records', indent=2)
                     logger.info(f"Ensemble predictions saved to {fmt.upper()}: {output_path}")
                 except Exception as e: logger.error(f"Failed to save ensemble predictions: {e}")
            else:
                print("\n--- Ensemble Strikeout Predictions ---")
                cols_to_print = ['gamePk', 'Team', 'Opponent', 'PitcherID', 'PitcherName', 'Predicted_SO_Ensemble']
                printable_cols = [col for col in cols_to_print if col in final_output_df.columns]
                print(final_output_df[printable_cols].to_string(index=False))
        else: logger.error("Final output DataFrame is empty for predictions.")

# --- Need updated parse_args ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate or Evaluate ENSEMBLE MLB Strikeout Predictions.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--prediction-date", type=str, help="Date (YYYY-MM-DD) to generate predictions for.")
    mode_group.add_argument("--evaluate-test-set", action="store_true", help="Evaluate model on historical test set.")
    parser.add_argument("--output-file", type=str, help="Optional path to save predictions/evaluation results.")
    parser.add_argument("--output-format", type=str, default="csv", choices=['csv', 'json'], help="Format for output file.")
    # Removed specific model/feature paths, they are found automatically now
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.prediction_date:
        try: datetime.strptime(args.prediction_date, "%Y-%m-%d")
        except ValueError: logger.error(f"Invalid format for --prediction-date: {args.prediction_date}. Use YYYY-MM-DD."); sys.exit(1)

    generate_ensemble_predictions_or_evaluate(args) # Call the main function
    logger.info("Ensemble script finished.")