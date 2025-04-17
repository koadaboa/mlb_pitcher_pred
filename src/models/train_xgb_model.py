# src/models/train_xgb_model.py

import pandas as pd
import numpy as np
import xgboost as xgb  # Use XGBoost
import optuna
import joblib
import pickle
import json
import argparse
import logging
import sys
import time
import re  # For parsing filenames
import os  # For path operations
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit  # Use TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Assuming script is run via python -m src.models.train_xgb_model
try:
    from src.config import DBConfig, StrikeoutModelConfig, LogConfig, FileConfig
    from src.data.utils import setup_logger, DBConnection, find_latest_file
    from src.features.selection import select_features # Import from shared location
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    # Add fallbacks if necessary for standalone testing
    MODULE_IMPORTS_OK = False
    sys.exit(1) # Exit if essential modules are missing

# Setup logger
LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True) # Ensure log dir exists
logger = setup_logger('train_xgb_model', LogConfig.LOG_DIR / 'train_xgb_model.log')

MODEL_TYPE = 'xgb' # Identifier for filenames

# --- Evaluation Metric Helper ---
def within_n_strikeouts(y_true, y_pred, n=1):
    """ Calculates percentage of predictions within n strikeouts """
    if y_true is None or y_pred is None or len(y_true) != len(y_pred): return np.nan
    y_true_arr = np.asarray(y_true); y_pred_arr = np.asarray(y_pred)
    return np.mean(np.abs(y_true_arr - np.round(y_pred_arr)) <= n)

# --- Optuna Objective Function (using TimeSeriesSplit for XGBoost) ---
def objective_xgb_timeseries_rmse(trial, X_data, y_data):
    """ Optuna objective function using TimeSeriesSplit CV for XGBoost """
    # Define search space for XGBoost
    xgb_params = {
        'objective': 'reg:squarederror', # Common regression objective
        'eval_metric': 'rmse',           # Evaluation metric
        'booster': 'gbtree',
        'n_jobs': -1, # Use all available cores
        'verbosity': 0, # Suppress verbose output during tuning
        'seed': StrikeoutModelConfig.RANDOM_STATE,
        # Tunable parameters
        'eta': trial.suggest_float('eta', 0.01, 0.2, log=True), # Learning rate
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), # Row subsampling
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # Feature subsampling
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5), # Min loss reduction for split
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True), # L2 regularization
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),    # L1 regularization
    }

    fold_rmses = []
    n_cv_splits = 4 # Number of time series splits
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)

    logger.debug(f"Trial {trial.number}: Starting TimeSeriesSplit CV with {n_cv_splits} splits...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_data)):
        X_train_fold, X_val_fold = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train_fold, y_val_fold = y_data.iloc[train_idx], y_data.iloc[val_idx]
        if len(X_val_fold) == 0: continue

        # Use XGBoost Scikit-Learn wrapper for Optuna integration with early stopping
        model = xgb.XGBRegressor(
            **xgb_params,
            n_estimators=1500, # High number for tuning, rely on early stopping
            early_stopping_rounds=50 # Early stopping within the fold
        )

        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  verbose=False) # Keep verbose False during tuning

        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        fold_rmses.append(rmse)
        logger.debug(f"  Trial {trial.number} Fold {fold+1}/{n_cv_splits} - Val RMSE: {rmse:.4f}")

    if not fold_rmses: return float('inf') # Return high error if CV failed
    average_rmse = np.mean(fold_rmses)
    logger.debug(f"Trial {trial.number} completed. Average CV RMSE: {average_rmse:.4f}")
    return average_rmse

# --- Argument Parser ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train XGBoost Strikeout Model")
    parser.add_argument("--optuna-trials", type=int, default=StrikeoutModelConfig.OPTUNA_TRIALS, help="Number of Optuna trials.")
    parser.add_argument("--optuna-timeout", type=int, default=StrikeoutModelConfig.OPTUNA_TIMEOUT, help="Optuna timeout in seconds.")
    parser.add_argument("--use-best-params", action="store_true",
                        help="Automatically load the latest saved best parameters for XGBoost (skips Optuna).")
    parser.add_argument("--production", action="store_true", help="Train final model on all data (no Optuna).")
    parser.add_argument("--final-estimators", type=int, default=StrikeoutModelConfig.FINAL_ESTIMATORS, help="Max estimators for final model.")
    parser.add_argument("--early-stopping", type=int, default=StrikeoutModelConfig.EARLY_STOPPING_ROUNDS, help="Early stopping rounds for final model.")
    parser.add_argument("--verbose-fit", action="store_true", help="Enable verbose fitting output.")
    parser.add_argument("--top-n-features", type=int, default=None,
                        help="If set, select and save only the top N features based on importance.")
    return parser.parse_args()

# --- Main Training Function ---
def train_model(args):
    """Loads data, trains XGBoost model (with optional Optuna), evaluates, and saves."""
    logger.info(f"Starting XGBoost training (Production Mode: {args.production})...")
    db_path = Path(DBConfig.PATH)
    model_dir = Path(FileConfig.MODELS_DIR)
    plot_dir = Path(FileConfig.PLOTS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Data (from _advanced tables) ---
    logger.info("Loading final feature data...")
    try:
        with DBConnection(db_path) as conn:
            train_df = pd.read_sql_query("SELECT * FROM train_features_advanced", conn)
            logger.info(f"Loaded {len(train_df)} train rows from 'train_features_advanced'")
            test_df = pd.read_sql_query("SELECT * FROM test_features_advanced", conn)
            logger.info(f"Loaded {len(test_df)} test rows from 'test_features_advanced'")
        all_data = pd.concat([train_df, test_df], ignore_index=True)
        logger.info(f"Total rows loaded: {len(all_data)}")
        logger.info(f"Available columns ({len(all_data.columns)}): {all_data.columns.tolist()}")
        del train_df, test_df; gc.collect()
    except Exception as e:
        logger.error(f"Error loading feature data: {e}", exc_info=True); sys.exit(1)

    # --- Feature Selection (Initial Exclusion using shared function) ---
    logger.info("Selecting features using shared function...")
    potential_feature_cols, _ = select_features(
        all_data.copy(),
        target_variable=StrikeoutModelConfig.TARGET_VARIABLE
    )
    if not potential_feature_cols:
        logger.error("No potential features selected after initial exclusion. Exiting.")
        sys.exit(1)
    logger.info(f"Found {len(potential_feature_cols)} potential features after initial exclusion.")

    # --- Load Correct Feature List (Full or Top-N) ---
    model_prefix = "prod_xgb" if args.production else "test_xgb" # Add xgb identifier
    logger.info("Searching for the latest saved XGBoost feature list file...")
    any_list_pattern = f"{model_prefix}_*_feature_columns_*.pkl"
    full_list_pattern = f"{model_prefix}_feature_columns_*.pkl" # Note: Ensure prefix matches saving convention
    latest_feature_file = find_latest_file(model_dir, any_list_pattern)

    if latest_feature_file:
        logger.info(f"Loading feature list from latest file: {latest_feature_file}")
        try:
            with open(latest_feature_file, 'rb') as f:
                training_features = pickle.load(f)
            logger.info(f"Loaded {len(training_features)} features from {latest_feature_file.name}")
        except Exception as e:
            logger.error(f"Failed to load feature list from {latest_feature_file}: {e}. Falling back.")
            training_features = potential_feature_cols
    else:
        logger.warning("No saved XGBoost feature list (.pkl) found. Using initially selected features.")
        training_features = potential_feature_cols

    # Final check: Ensure loaded/selected features exist in the data
    missing_in_data = [f for f in training_features if f not in all_data.columns]
    if missing_in_data:
        logger.error(f"Features in loaded list missing from data: {missing_in_data}.")
        training_features = [f for f in training_features if f in all_data.columns] # Use only available
        if not training_features:
             logger.error("No valid features remain after checking data. Exiting.")
             sys.exit(1)
    logger.info(f"Using {len(training_features)} features for this XGBoost training run.")

    # --- Prepare Data Splits ---
    train_indices = all_data[all_data['season'].isin(StrikeoutModelConfig.DEFAULT_TRAIN_YEARS)].index
    test_indices = all_data[all_data['season'].isin(StrikeoutModelConfig.DEFAULT_TEST_YEARS)].index

    X_train_full = all_data.loc[train_indices, training_features].copy()
    y_train_full = all_data.loc[train_indices, StrikeoutModelConfig.TARGET_VARIABLE].copy()
    X_test = all_data.loc[test_indices, training_features].copy()
    y_test = all_data.loc[test_indices, StrikeoutModelConfig.TARGET_VARIABLE].copy()

    # --- Handle Infinities/NaNs ---
    logger.info("Checking for infinite/NaN values before training...")
    for df_name, df_check in [('X_train_full', X_train_full), ('X_test', X_test)]:
        if not df_check.empty:
            inf_mask = np.isinf(df_check).any().any()
            nan_mask = df_check.isnull().any().any()
            if inf_mask or nan_mask:
                 logger.warning(f"Infinite or NaN values found in {df_name}. Replacing Inf->NaN, then imputing NaNs with train median.")
                 df_check.replace([np.inf, -np.inf], np.nan, inplace=True)
                 for col in df_check.columns[df_check.isnull().any()]:
                      median_val = X_train_full[col].median() # Use train median for consistency
                      fill_val = median_val if pd.notna(median_val) else 0
                      df_check[col].fillna(fill_val, inplace=True)
                      # logger.debug(f"Imputed NaNs in {df_name} column '{col}' with train median {fill_val:.4f}")

    logger.info(f"Train shape: {X_train_full.shape}, Test shape: {X_test.shape}")
    del all_data; gc.collect()

    # --- Hyperparameter Loading / Optuna ---
    best_params = None
    should_run_optuna = False
    param_pattern = f"{model_prefix}_best_params_*.json" # Use specific prefix

    if args.production:
        logger.info("--- Running in PRODUCTION MODE (XGBoost) ---")
        logger.info("Attempting to load latest best parameters for production...")
        should_run_optuna = False
        latest_params_file = find_latest_file(model_dir, param_pattern)
        if latest_params_file:
            logger.info(f"Loading hyperparameters from: {latest_params_file}")
            try:
                with open(latest_params_file, 'r') as f: best_params = json.load(f)
                logger.info(f"Successfully loaded XGBoost parameters for production.")
            except Exception as e: logger.error(f"Failed load prod params: {e}. Using defaults."); best_params = {}
        else: logger.warning(f"No XGBoost param file found for production. Using defaults."); best_params = {}

    elif args.use_best_params:
        logger.info("Attempting load latest best XGBoost params (--use-best-params specified)...")
        should_run_optuna = False
        latest_params_file = find_latest_file(model_dir, param_pattern)
        if latest_params_file:
            logger.info(f"Loading hyperparameters from: {latest_params_file}")
            try:
                with open(latest_params_file, 'r') as f: best_params = json.load(f)
                logger.info(f"Successfully loaded XGBoost parameters.")
            except Exception as e: logger.error(f"Failed load params: {e}. Run Optuna."); best_params=None; should_run_optuna=True
        else: logger.warning(f"No XGBoost param file found. Run Optuna."); best_params=None; should_run_optuna=True
    else:
        logger.info("No parameters loaded or specified (--use-best-params not set). Running Optuna for XGBoost.")
        should_run_optuna = True; best_params = None

    # Run Optuna if needed
    if should_run_optuna:
        logger.info("Running Optuna hyperparameter search for XGBoost...")
        optuna_start_time = time.time()
        study = optuna.create_study(direction='minimize')
        try:
            # Pass FULL training data for TimeSeriesSplit CV
            study.optimize(lambda trial: objective_xgb_timeseries_rmse(trial, X_train_full, y_train_full),
                           n_trials=args.optuna_trials, timeout=args.optuna_timeout)
            best_params = study.best_params
            logger.info(f"Optuna finished in {(time.time() - optuna_start_time):.2f}s. Best Avg CV RMSE: {study.best_value:.4f}")
            logger.info(f"Best XGBoost hyperparameters: {best_params}")
            # Save the new best parameters
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            params_save_path = model_dir / f"{model_prefix}_best_params_{timestamp}.json"
            try:
                with open(params_save_path, 'w') as f: json.dump(best_params, f, indent=4)
                logger.info(f"Saved best hyperparameters to: {params_save_path}")
            except Exception as e: logger.error(f"Failed to save best params: {e}")
        except Exception as e:
             logger.error(f"Optuna optimization failed: {e}", exc_info=True)
             logger.warning("Proceeding with default XGBoost parameters.")
             best_params = {} # Fallback

    # Ensure defaults if needed
    if best_params is None: best_params = {}
    if not best_params: # If empty dict after all attempts
        logger.warning("No valid XGBoost hyperparameters. Using default parameters.")
        # Define XGBoost defaults (example)
        best_params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'eta': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3, 'seed': StrikeoutModelConfig.RANDOM_STATE}
    # Ensure core params exist
    best_params.setdefault('objective', 'reg:squarederror')
    best_params.setdefault('eval_metric', 'rmse')
    best_params.setdefault('seed', StrikeoutModelConfig.RANDOM_STATE)
    best_params.setdefault('n_jobs', -1)

    # --- Final Model Training ---
    logger.info("Training final XGBoost model on full training data...")
    final_model = xgb.XGBRegressor(
        **best_params,
        n_estimators=args.final_estimators,
        early_stopping_rounds=args.early_stopping,
        # seed/n_jobs already in best_params
    )
    eval_set = [(X_test, y_test)] if not X_test.empty and not y_test.empty else None

    fit_verbose = 100 if args.verbose_fit else False # Control verbosity level for XGBoost

    try:
        final_model.fit(X_train_full, y_train_full,
                        eval_set=eval_set,
                        verbose=fit_verbose)
        logger.info("Final XGBoost model training complete.")
        if hasattr(final_model, 'best_iteration') and final_model.best_iteration is not None:
             logger.info(f"Best iteration: {final_model.best_iteration}")
        # Note: final_model.best_score might also be available depending on setup
    except Exception as e:
        logger.error(f"Failed to train final XGBoost model: {e}", exc_info=True); sys.exit(1)

    # --- Evaluation ---
    logger.info("Evaluating XGBoost model...")
    train_preds = final_model.predict(X_train_full)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))
    train_mae = mean_absolute_error(y_train_full, train_preds)

    logger.info("--- Train Metrics (XGBoost) ---")
    logger.info(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")

    if not X_test.empty and not y_test.empty:
        test_preds = final_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_w1 = within_n_strikeouts(y_test, test_preds, n=1)
        test_w2 = within_n_strikeouts(y_test, test_preds, n=2)
        logger.info("--- Test Metrics (XGBoost) ---")
        logger.info(f"Test RMSE : {test_rmse:.4f}, Test MAE : {test_mae:.4f}")
        logger.info(f"Test W/1 K: {test_w1:.4f}, Test W/2 K: {test_w2:.4f}")
    else:
        logger.info("Test set not available for evaluation.")

    # --- Save Artifacts ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_prefix already defined based on production/test mode

    # Feature Importance
    try:
        importances = final_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': training_features, # Use the list of features model was trained on
            'importance': importances
        }).sort_values('importance', ascending=False)
        imp_path = model_dir / f"{model_prefix}_feature_importance_full_{timestamp}.csv"
        importance_df.to_csv(imp_path, index=False)
        logger.info(f"Full XGBoost importance list saved: {imp_path}")
        logger.info("Top 20 XGBoost Feature Importances:")
        logger.info("\n" + importance_df.head(20).to_string(index=False))
    except Exception as e:
        logger.error(f"Could not get/save XGBoost feature importances: {e}")

    # Importance Plot
    try:
        plt.figure(figsize=(10, 8))
        sns.barplot(x="importance", y="feature", data=importance_df.head(30))
        plt.title(f"XGBoost Feature Importance ({model_prefix.replace('_xgb','')})") # Clean title
        plt.tight_layout()
        plot_path = plot_dir / f"{model_prefix}_feat_imp_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"XGBoost importance plot: {plot_path}")
        plt.close()
    except Exception as e: logger.error(f"Failed to create XGBoost importance plot: {e}")

    # Prediction vs Actual Plot
    try:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, test_preds, alpha=0.3, label='Test Set')
        plt.scatter(y_train_full, train_preds, alpha=0.05, label='Train Set', color='orange')
        min_val = min(min(y_test), min(y_train_full), 0) if not y_test.empty else 0
        max_val = max(max(y_test), max(y_train_full)) if not y_test.empty else max(y_train_full)
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label='Perfect Prediction')
        plt.xlabel("Actual Strikeouts"); plt.ylabel("Predicted Strikeouts")
        plt.title(f"Actual vs. Predicted Strikeouts (XGBoost - {model_prefix.replace('_xgb','')})")
        plt.legend(); plt.grid(True)
        plot_path = plot_dir / f"{model_prefix}_pred_actual_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"XGBoost Pred/Actual plot: {plot_path}")
        plt.close()
    except Exception as e: logger.error(f"Failed to create XGBoost pred/actual plot: {e}")

    # Save Model (using .json for portability or .ubj for efficiency)
    model_path = model_dir / f"{model_prefix}_strikeout_model_{timestamp}.json"
    try:
        final_model.save_model(str(model_path))
        logger.info(f"XGBoost model saved: {model_path}")
    except Exception as e: logger.error(f"Failed to save XGBoost model: {e}")

    # --- Save Feature List (Full or Top-N) ---
    if args.top_n_features is not None and args.top_n_features > 0:
        if args.top_n_features < len(importance_df):
            top_n_features_list = importance_df['feature'].head(args.top_n_features).tolist()
            logger.info(f"Selecting top {args.top_n_features} XGBoost features based on importance.")
            top_n_features_path = model_dir / f"{model_prefix}_top_{args.top_n_features}_feature_columns_{timestamp}.pkl"
            try:
                with open(top_n_features_path, 'wb') as f: pickle.dump(top_n_features_list, f)
                logger.info(f"Top {args.top_n_features} XGBoost features list saved: {top_n_features_path}")
            except Exception as e: logger.error(f"Failed to save top N XGBoost feature list: {e}")
        else:
            logger.warning(f"--top-n-features ({args.top_n_features}) >= total features ({len(importance_df)}). Saving full list.")
            features_path = model_dir / f"{model_prefix}_feature_columns_{timestamp}.pkl"
            try:
                with open(features_path, 'wb') as f: pickle.dump(training_features, f)
                logger.info(f"Full XGBoost features list saved: {features_path}")
            except Exception as e: logger.error(f"Failed to save full XGBoost feature list: {e}")
    else:
        # Save the full feature list used in this run
        features_path = model_dir / f"{model_prefix}_feature_columns_{timestamp}.pkl"
        try:
            with open(features_path, 'wb') as f: pickle.dump(training_features, f)
            logger.info(f"XGBoost features list used for this run saved: {features_path}")
        except Exception as e: logger.error(f"Failed to save XGBoost feature list: {e}")

    # Save best params (if found via Optuna)
    if should_run_optuna and best_params and study.best_value is not None:
         params_save_path_final = model_dir / f"{model_prefix}_best_params_{timestamp}.json"
         try:
             params_to_save = best_params.copy()
             params_to_save['best_iteration_from_final_train'] = final_model.best_iteration # Iteration from final train
             params_to_save['best_cv_score_from_optuna'] = study.best_value
             with open(params_save_path_final, 'w') as f: json.dump(params_to_save, f, indent=4)
             logger.info(f"Saved final XGBoost hyperparameters (from Optuna) to: {params_save_path_final}")
         except Exception as e: logger.error(f"Failed to save Optuna best parameters: {e}")
    elif best_params:
         logger.info(f"Final XGBoost model trained using loaded parameters.")

    logger.info("XGBoost model training finished successfully.")


# --- Main Execution Block ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    args = parse_args()
    train_model(args)
    logger.info("--- XGBoost Training Script Completed ---")