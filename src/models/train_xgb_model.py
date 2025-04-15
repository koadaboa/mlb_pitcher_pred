# src/models/train_xgb_model.py
import pandas as pd
import numpy as np
import xgboost as xgb # Changed import
import pickle
import logging
import argparse
from pathlib import Path
import sys
import json # For saving params
import optuna # For hyperparameter tuning
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split # For creating validation set
from datetime import datetime

# Ensure src directory is in the path if running script directly
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Imports ---
# (Imports will fail immediately if modules are missing)
from src.data.utils import setup_logger, DBConnection
from src.config import DBConfig, StrikeoutModelConfig # Assuming StrikeoutModelConfig holds train/test years

# --- Logger Setup ---
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True, parents=True)
logger = setup_logger('train_xgb_model', log_file= log_dir / 'train_xgb_model.log', level=logging.INFO)

# --- Constants ---
TARGET_VARIABLE = 'strikeouts'
MODEL_TYPE = 'xgboost' # Identifier for filenames

# --- Data Loading ---
def load_features(db_path, table_name):
    """Loads features from the specified table."""
    logger.info(f"Loading features from table: {table_name}")
    try:
        with DBConnection(db_path) as conn:
             if conn is None: raise ConnectionError("DB Connection failed.")
             cursor = conn.cursor()
             cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
             if not cursor.fetchone():
                  logger.error(f"Feature table '{table_name}' not found. Run engineer_features.py first.")
                  return None
             features_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
             if features_df.empty:
                  logger.warning(f"No data found in '{table_name}'.")
                  return None # Return None for empty
             logger.info(f"Loaded {len(features_df)} rows from {table_name}.")
             # Basic type conversion/cleanup if needed
             features_df['game_date'] = pd.to_datetime(features_df['game_date'], errors='coerce')
             # Convert potential Int64/int32 to standard int64 for XGBoost if needed
             for col in features_df.select_dtypes(include=['Int64', 'int32']).columns:
                  # Check if conversion is necessary to avoid unnecessary warnings/operations
                  if features_df[col].dtype != 'int64':
                       features_df[col] = features_df[col].astype('int64')
             return features_df
    except Exception as e:
        logger.error(f"Error loading features from {table_name}: {e}", exc_info=True)
        return None

# --- Feature Selection ---
def select_features(df):
    """Selects numeric features, excluding specified columns."""
    if df is None or df.empty:
        return [], df

    exclude_cols = [
        'index', '', 'pitcher_id', 'player_name', 'game_pk', 'home_team', 'away_team',
        'opponent_team_name', 'game_date', 'season', 'game_month', 'year',
        'p_throws', 'stand', 'team', 'Team', 'opp_base_team', 'opp_adv_team', 'opp_adv_opponent',
        'strikeouts', # Target
        # Potential leakage columns (ensure consistency)
        'k_per_9', 'k_percent', 'batters_faced', 'total_pitches', 'avg_velocity',
        'max_velocity', 'avg_spin_rate', 'avg_horizontal_break', 'avg_vertical_break',
        'zone_percent', 'swinging_strike_percent', 'innings_pitched', 'fastball_percent',
        'breaking_percent', 'offspeed_percent', 'inning', 'score_differential',
        'is_close_game', 'is_playoff',
        # Low importance features (from previous analysis)
        'is_home', 'rest_days_6_more', 'rest_days_4_less',
        'is_month_3', 'is_month_4', 'is_month_5', 'is_month_6', 'is_month_7', 'is_month_8', 'is_month_9', 'is_month_10',
        'rest_days_5', 'throws_right'
        # Add any others you identified
        'rolling_5g_fastball_pct', 'rolling_3g_fastball_pct', 'rolling_5g_k9', 'ewma_10g_avg_velocity_lag1',
        'rolling_10g_velocity', 'rolling_3g_breaking_pct', 'ewma_5g_k_per_9_lag1', 'ewma_5g_k_percent_lag1',
        'rolling_3g_breaking_percent_std_lag1', 'rolling_10g_k9'
    ]

    # Select numeric columns only, excluding the ones in the list
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    logger.info(f"Selected {len(feature_cols)} numeric features.")
    logger.debug(f"Excluded columns: {exclude_cols}")
    logger.debug(f"First 5 selected features: {feature_cols[:5]}")

    # Ensure TARGET_VARIABLE exists before trying to select it
    columns_to_return = feature_cols + ([TARGET_VARIABLE] if TARGET_VARIABLE in df.columns else [])
    return feature_cols, df[columns_to_return].copy() # Return a copy

# --- Optuna Objective Function ---
def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for XGBoost hyperparameter tuning."""
    # Define XGBoost hyperparameter search space
    param = {
        'objective': 'reg:squarederror', # Regression task
        'eval_metric': 'rmse',           # Evaluation metric
        'booster': 'gbtree',
        'n_jobs': -1,                    # Use all available cores
        'verbosity': 0,                  # Suppress XGBoost messages during tuning
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True), # Learning rate
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), # Fraction of samples used per tree
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # Fraction of features used per tree
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5), # Min loss reduction to make split
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True), # L2 regularization
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),   # L1 regularization
    }

    # Use XGBoost Scikit-learn wrapper
    model = xgb.XGBRegressor(**param,
                             n_estimators=1000,
                             early_stopping_rounds=50,
                             random_state=42)

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    return rmse

# --- Main Training Function ---
def train_model(args):
    """Loads data, selects features, tunes hyperparameters (optional), trains, evaluates, and saves."""
    db_path = project_root / DBConfig.PATH
    model_dir = project_root / 'models'
    model_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Load Data
    train_df_raw = load_features(db_path, 'train_combined_features')
    test_df_raw = load_features(db_path, 'test_combined_features')

    if train_df_raw is None or train_df_raw.empty: logger.error("Training data failed to load or is empty."); return
    if test_df_raw is None or test_df_raw.empty: logger.warning("Test data failed to load or is empty."); test_df_raw = pd.DataFrame()

    # 2. Select Features
    feature_cols, train_df_selected = select_features(train_df_raw)
    if not feature_cols: logger.error("No features selected."); return

    # Prepare test set
    if not test_df_raw.empty:
        test_cols_needed = feature_cols + [TARGET_VARIABLE]
        test_cols_avail = [col for col in test_cols_needed if col in test_df_raw.columns]
        if len(test_cols_avail) != len(test_cols_needed): logger.warning(f"Test set missing columns.");
        test_df_selected = test_df_raw[test_cols_avail].copy()
        if TARGET_VARIABLE not in test_df_selected.columns: logger.warning(f"Target missing from test set."); test_df_selected = pd.DataFrame()
    else:
        test_df_selected = pd.DataFrame()

    # Prepare final X, y
    X_train_full = train_df_selected[feature_cols]
    y_train_full = train_df_selected[TARGET_VARIABLE]

    X_test = pd.DataFrame(); y_test = pd.Series(dtype='float64')
    if not test_df_selected.empty:
        X_test = test_df_selected[feature_cols]
        y_test = test_df_selected[TARGET_VARIABLE]

    # --- *** NEW: Check for and handle infinite values *** ---
    logger.info("Checking for infinite values in feature sets...")
    data_frames_to_check = {'X_train_full': X_train_full, 'X_test': X_test}
    for df_name, df_check in data_frames_to_check.items():
        if df_check.empty: continue
        inf_mask = np.isinf(df_check)
        if inf_mask.any().any():
            inf_cols = df_check.columns[inf_mask.any()].tolist()
            logger.warning(f"Infinite values found in {df_name} columns: {inf_cols}. Replacing with NaN.")
            # Replace inf with NaN
            df_check.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Optional: Log how many were replaced per column
            # logger.info(f"NaN counts after replacing inf in {df_name}:\n{df_check[inf_cols].isnull().sum()}")
        else:
            logger.info(f"No infinite values found in {df_name}.")
    # --- *** END CHECK *** ---


    # 3. Hyperparameter Tuning (Optuna) or Load Params
    best_params = None
    if args.params_file and Path(args.params_file).exists():
        logger.info(f"Loading hyperparameters from: {args.params_file}")
        try:
            with open(args.params_file, 'r') as f: best_params = json.load(f)
        except Exception as e: logger.error(f"Failed load params file: {e}. Running Optuna.")
    # --- Run Optuna if no params file provided ---
    if best_params is None: # Check if still None (i.e., no file or loading failed)
        logger.info("Running Optuna hyperparameter search...")
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42
        )
        logger.info(f"Optuna train size: {len(X_train_opt)}, Val size: {len(X_val_opt)}")
        study = optuna.create_study(direction='minimize')
        try:
            study.optimize(lambda trial: objective(trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt),
                           n_trials=args.optuna_trials, timeout=args.optuna_timeout)
            best_params = study.best_params
            logger.info(f"Optuna finished. Best RMSE: {study.best_value:.4f}")
            logger.info(f"Best hyperparameters: {best_params}")
        except Exception as e:
             logger.error(f"Optuna optimization failed: {e}", exc_info=True)
             logger.warning("Proceeding with default XGBoost parameters.")
             best_params = {}

        # Save best params
        params_filename = model_dir / f'{MODEL_TYPE}_best_params_{timestamp}.json'
        try:
            with open(params_filename, 'w') as f: json.dump(best_params, f, indent=4)
            logger.info(f"Saved best hyperparameters to: {params_filename}")
        except Exception as e: logger.error(f"Failed save best params: {e}")

    # Use default params if tuning wasn't run or failed and no file loaded
    if best_params is None: best_params = {}
    if 'objective' not in best_params: best_params['objective'] = 'reg:squarederror'
    if 'eval_metric' not in best_params: best_params['eval_metric'] = 'rmse'


    # 4. Train Final Model
    logger.info("Training final XGBoost model on full training data...")
    final_model = xgb.XGBRegressor(**best_params,
                                   n_estimators=args.final_estimators,
                                   early_stopping_rounds=args.early_stopping,
                                   random_state=42,
                                   n_jobs=-1)
    eval_set = []
    if not X_test.empty and not y_test.empty:
        eval_set = [(X_test, y_test)]
        logger.info("Using test set for final model early stopping.")
    else: logger.warning("No test set available for final model early stopping.")

    try:
        final_model.fit(X_train_full, y_train_full, eval_set=eval_set, verbose=args.verbose_fit)
        logger.info("Final model training complete.")
        if hasattr(final_model, 'best_iteration') and final_model.best_iteration is not None:
             logger.info(f"Best iteration: {final_model.best_iteration}")
    except Exception as e: logger.error(f"Failed to train final XGBoost model: {e}", exc_info=True); return

    # 5. Evaluate Model
    logger.info("Evaluating model...")
    # ... (Evaluation logic - same as before) ...
    train_preds = final_model.predict(X_train_full)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))
    train_mae = mean_absolute_error(y_train_full, train_preds)
    logger.info(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
    if not X_test.empty and not y_test.empty:
        test_preds = final_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        logger.info(f"Test RMSE : {test_rmse:.4f}, Test MAE : {test_mae:.4f}")
    else: logger.info("Test set not available for evaluation.")


    # 6. Feature Importance
    # ... (Feature importance logic - same as before) ...
    try:
        importances = final_model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': feature_cols,'importance': importances}).sort_values(by='importance', ascending=False)
        logger.info("Top 20 Feature Importances:"); logger.info("\n" + feature_importance_df.head(20).to_string(index=False))
        importance_filename = model_dir / f'{MODEL_TYPE}_feature_importance_{timestamp}.csv'
        feature_importance_df.to_csv(importance_filename, index=False)
        logger.info(f"Feature importances saved to: {importance_filename}")
    except Exception as e: logger.error(f"Could not get/save feature importances: {e}")


    # 7. Save Model, Features List
    # ... (Saving logic - same as before) ...
    model_filename = model_dir / f'{MODEL_TYPE}_strikeout_model_{timestamp}.json'
    try: final_model.save_model(str(model_filename)); logger.info(f"Model saved to: {model_filename}")
    except Exception as e: logger.error(f"Failed save model: {e}", exc_info=True)
    features_filename = model_dir / f'{MODEL_TYPE}_feature_columns_{timestamp}.pkl'
    try:
        with open(features_filename, 'wb') as f: pickle.dump(feature_cols, f)
        logger.info(f"Feature list saved to: {features_filename}")
    except Exception as e: logger.error(f"Failed save feature list: {e}", exc_info=True)

    logger.info("Model training script finished.")


# --- Argument Parser and Main Execution Block ---
# ... (Argument parsing and main block - same as before) ...
def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost Strikeout Prediction Model.")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--optuna-timeout", type=int, default=3600, help="Optuna timeout in seconds.")
    parser.add_argument("--params-file", type=str, default=None, help="Path to JSON file with pre-defined XGBoost parameters (skips Optuna).")
    parser.add_argument("--final-estimators", type=int, default=1000, help="Max number of estimators for final model training.")
    parser.add_argument("--early-stopping", type=int, default=50, help="Early stopping rounds for final model training.")
    parser.add_argument("--verbose-fit", action="store_true", help="Make XGBoost fitting verbose.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Starting XGBoost model training with args: {args}")
    train_model(args)
    logger.info("--- XGBoost Training Script Completed ---")
