# src/scripts/predict_today.py

import pandas as pd
import numpy as np
import pickle
import joblib # For loading sklearn models/scalers (like ElasticNet)
import lightgbm as lgb # For loading LGBM models
import xgboost as xgb # For loading XGB models
import json
import argparse
import logging
import sys
import subprocess  # run feature pipeline
from datetime import datetime
from pathlib import Path

# Assuming script is run via python -m src.scripts.predict_today
try:
    from src.config import DBConfig, StrikeoutModelConfig, LogConfig, FileConfig
    from src.data.utils import setup_logger, DBConnection, find_latest_file # Import find_latest_file
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    MODULE_IMPORTS_OK = False
    sys.exit(1)

# Setup logger
LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger('predict_today', LogConfig.LOG_DIR / 'predict_today.log')

# --- Argument Parser ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate MLB Pitcher Strikeout Predictions for a specific date.")
    parser.add_argument("--prediction-date", type=str, required=True,
                        help="The date for which to generate predictions (YYYY-MM-DD).")
    parser.add_argument("--model-type", type=str, required=True, choices=['lgb', 'xgb', 'enet'],
                        help="Specify the type of model artifacts to load ('lgb', 'xgb', 'enet').")
    parser.add_argument("--top-n-features", type=int, default=None,
                        help="Specify N to load the latest 'top_N' feature list. If omitted, loads the latest full feature list.")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Optional path to save predictions CSV/JSON file.")
    parser.add_argument("--output-format", type=str, default="csv", choices=['csv', 'json'],
                        help="Format for the output file (csv or json). Default: csv")
    parser.add_argument("--production-model", action="store_true",
                        help="Load the latest 'prod' model artifacts instead of 'test'.")

    # Remove old arguments:
    # parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model file.")
    # parser.add_argument("--features-path", type=str, required=True, help="Path to the feature list file (.pkl).")

    return parser.parse_args()

# --- Main Prediction Function ---
def predict_strikeouts(args):
    """Loads data, finds artifacts, generates predictions, and saves results."""
    logger.info(f"Starting predictions for date: {args.prediction_date} using model type: {args.model_type}")
    # Ensure features are generated for this date
    logger.info(f"Running feature pipeline for date: {args.prediction_date}")
    try:
        subprocess.check_call([sys.executable, "-m", "src.scripts.run_feature_pipeline", "--prediction-date", args.prediction_date])
    except subprocess.CalledProcessError as e:
        logger.error(f"Feature pipeline failed: {e}")
        sys.exit(e.returncode)

    db_path = Path(DBConfig.PATH)
    model_dir = Path(FileConfig.MODELS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True) # Ensure model dir exists

    model_prefix = f"prod_{args.model_type}" if args.production_model else f"test_{args.model_type}"
    logger.info(f"Using model prefix: {model_prefix}")

    # --- Find Latest Artifacts ---
    logger.info("Searching for latest model artifacts...")

    # 1. Find Feature List (.pkl)
    feature_list_path = None
    if args.top_n_features:
        logger.info(f"Searching for latest top-{args.top_n_features} feature list...")
        top_n_pattern = f"{model_prefix}_top_{args.top_n_features}_feature_columns_*.pkl"
        feature_list_path = find_latest_file(model_dir, top_n_pattern)
        if not feature_list_path:
            logger.warning(f"No top-{args.top_n_features} feature list found. Falling back to latest full list.")

    if not feature_list_path:
        logger.info(f"Searching for latest full feature list for {args.model_type}...")
        full_list_pattern = f"{model_prefix}_feature_columns_*.pkl"
        feature_list_path = find_latest_file(model_dir, full_list_pattern)

    if not feature_list_path:
        logger.error(f"CRITICAL: Could not find any feature list file (.pkl) for prefix '{model_prefix}'. Ensure models have been trained.")
        sys.exit(1)

    # 2. Find Model File
    model_file_path = None
    if args.model_type == 'lgb':
        model_pattern = f"{model_prefix}_strikeout_model_*.txt"
        model_file_path = find_latest_file(model_dir, model_pattern)
    elif args.model_type == 'xgb':
        # Prefer .json, fallback to .ubj if needed
        model_pattern_json = f"{model_prefix}_strikeout_model_*.json"
        model_pattern_ubj = f"{model_prefix}_strikeout_model_*.ubj"
        model_file_path = find_latest_file(model_dir, model_pattern_json)
        if not model_file_path:
             model_file_path = find_latest_file(model_dir, model_pattern_ubj)
    elif args.model_type == 'enet':
        model_pattern = f"{model_prefix}_strikeout_model_*.joblib"
        model_file_path = find_latest_file(model_dir, model_pattern)

    if not model_file_path:
        logger.error(f"CRITICAL: Could not find any model file for prefix '{model_prefix}'. Ensure models have been trained.")
        sys.exit(1)

    # 3. Find Scaler File (only if needed)
    scaler_path = None
    scaler = None
    if args.model_type in ['enet']: # Add other models requiring scaling here (e.g., 'svr')
        logger.info(f"Searching for latest scaler for {args.model_type}...")
        scaler_pattern = f"{model_prefix}_scaler_*.joblib"
        scaler_path = find_latest_file(model_dir, scaler_pattern)
        if not scaler_path:
            logger.error(f"CRITICAL: Model type '{args.model_type}' requires a scaler, but scaler file not found for prefix '{model_prefix}'.")
            sys.exit(1)

    # --- Load Artifacts ---
    logger.info(f"Loading feature list from: {feature_list_path}")
    try:
        with open(feature_list_path, 'rb') as f:
            feature_columns = pickle.load(f)
        logger.info(f"Loaded {len(feature_columns)} features.")
    except Exception as e:
        logger.error(f"Failed to load feature list: {e}", exc_info=True); sys.exit(1)

    logger.info(f"Loading model from: {model_file_path}")
    model = None
    try:
        if args.model_type == 'lgb':
            model = lgb.Booster(model_file=str(model_file_path))
        elif args.model_type == 'xgb':
            # XGBoost loading depends on how it was saved. Assume Booster for now.
            # If saved via Scikit-Learn wrapper's save_model:
            model = xgb.Booster()
            model.load_model(str(model_file_path))
            # If saved via joblib (less common for Booster):
            # model = joblib.load(model_file_path)
        elif args.model_type == 'enet':
            model = joblib.load(model_file_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True); sys.exit(1)

    if scaler_path:
        logger.info(f"Loading scaler from: {scaler_path}")
        try:
            scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}", exc_info=True); sys.exit(1)


    # --- Load Prediction Data ---
    logger.info(f"Loading prediction features for date: {args.prediction_date}")
    try:
        with DBConnection(db_path) as conn:
            # Load features AND identifying columns
            query = f"SELECT * FROM prediction_features_advanced WHERE DATE(game_date) = '{args.prediction_date}'"
            pred_data = pd.read_sql_query(query, conn)
        if pred_data.empty:
            logger.error(f"No prediction data found for date {args.prediction_date} in 'prediction_features_advanced'. Ensure engineer_features ran correctly.")
            sys.exit(1)
        logger.info(f"Loaded {len(pred_data)} rows for prediction.")
    except Exception as e:
        logger.error(f"Failed to load prediction data: {e}", exc_info=True); sys.exit(1)

    # --- Prepare Features for Prediction ---
    # Check if all required features are present in the prediction data
    missing_features = [f for f in feature_columns if f not in pred_data.columns]
    if missing_features:
        logger.error(f"Features required by the model are missing from the prediction data: {missing_features}")
        logger.error("Cannot proceed with prediction. Check feature engineering step.")
        sys.exit(1)

    # --- *** VERIFY Feature List Correctness *** ---
    # Use TARGET_ENCODING_COLS imported from config
    original_categorical_cols = StrikeoutModelConfig.TARGET_ENCODING_COLS

    # Check if the feature_columns list loaded from the file accidentally contains originals
    originals_in_feature_list = [col for col in original_categorical_cols if col in feature_columns]
    if originals_in_feature_list:
         logger.error(f"CRITICAL: The loaded feature list ({feature_list_path.name}) contains original categorical columns that should have been encoded: {originals_in_feature_list}")
         logger.error(f"Model expects encoded features only. Ensure the feature list saved during training (e.g., in {model_dir}) is correct.")
         sys.exit(1)
    else:
         logger.info("Verified: Loaded feature list does not contain original categorical columns from config.")
    # --- *** END VERIFICATION *** ---

    X_pred = pred_data[feature_columns].copy()

    # Handle potential NaNs/Infs in prediction data (impute with 0 or a robust strategy)
    if X_pred.isnull().any().any() or np.isinf(X_pred.to_numpy()).any():
        logger.warning("NaN or Inf values found in prediction features. Imputing with 0.")
        X_pred.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_pred.fillna(0, inplace=True) # Simple imputation with 0 for prediction

    # Scale features if scaler was loaded
    if scaler:
        logger.info("Scaling prediction features...")
        try:
            X_pred_scaled = scaler.transform(X_pred)
            # Maintain DataFrame structure if possible (depends on scaler output)
            if isinstance(X_pred_scaled, np.ndarray):
                 X_pred = pd.DataFrame(X_pred_scaled, index=X_pred.index, columns=X_pred.columns)
            else: # Assume it returned a DataFrame
                 X_pred = X_pred_scaled
        except Exception as e:
            logger.error(f"Error scaling prediction data: {e}", exc_info=True); sys.exit(1)

    # --- Generate Predictions ---
    logger.info("Generating predictions...")
    try:
        predictions = model.predict(X_pred)
        # Ensure predictions are non-negative for applicable models
        if args.model_type in ['enet']: # Add other models if needed
             predictions = np.maximum(0, predictions)
        logger.info("Predictions generated successfully.")
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True); sys.exit(1)

    # --- Format Output ---
    output_df = pred_data[['game_pk', 'game_date', 'pitcher_id', 'player_name', 'team', 'opponent_team', 'is_home']].copy()
    output_df['predicted_strikeouts'] = predictions
    # Optional: Round predictions for display
    output_df['predicted_strikeouts_rounded'] = np.round(predictions).astype(int)

    logger.info("--- Prediction Summary ---")
    logger.info(f"\n{output_df.to_string(index=False)}")
    logger.info("--------------------------")

    # --- Save Output ---
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        try:
            if args.output_format == 'csv':
                output_df.to_csv(output_path, index=False)
                logger.info(f"Predictions saved to CSV: {output_path}")
            elif args.output_format == 'json':
                output_df.to_json(output_path, orient='records', indent=4)
                logger.info(f"Predictions saved to JSON: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save predictions to {output_path}: {e}")

    logger.info("Prediction script finished successfully.")


# --- Main Execution Block ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    args = parse_args()
    predict_strikeouts(args)
    logger.info("--- Prediction Script Completed ---")
