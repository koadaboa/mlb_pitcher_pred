# src/models/train_lgb_model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib
import pickle
import json
import argparse
import logging
import sys
import time
import re
import os
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Assuming script is run via python -m src.models.train_lgb_model
try:
    from src.config import DBConfig, StrikeoutModelConfig, LogConfig, FileConfig
    # Import find_latest_file from utils
    from src.data.utils import setup_logger, DBConnection, find_latest_file
    from src.features.selection import select_features # Import from shared location
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    MODULE_IMPORTS_OK = False
    sys.exit(1) # Exit if essential modules are missing

# Setup logger
LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True) # Ensure log dir exists
logger = setup_logger('train_lgb_model', LogConfig.LOG_DIR / 'train_lgb_model.log')

MODEL_TYPE = 'lgb' # Identifier for filenames

# --- Evaluation Metric Helper ---
def within_n_strikeouts(y_true, y_pred, n=1):
    """ Calculates percentage of predictions within n strikeouts """
    if y_true is None or y_pred is None or len(y_true) != len(y_pred): return np.nan
    y_true_arr = np.asarray(y_true); y_pred_arr = np.asarray(y_pred)
    return np.mean(np.abs(y_true_arr - np.round(y_pred_arr)) <= n)

# --- Optuna Objective Function (using TimeSeriesSplit) ---
def objective_lgbm_timeseries_rmse(trial, X_data, y_data):
    """ Optuna objective function using TimeSeriesSplit CV """
    # Define search space for LightGBM
    lgb_params = {
        'objective': 'regression_l1', # MAE objective
        'metric': 'rmse',             # Report RMSE
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'verbose': -1, # Suppress verbose output during tuning
        'seed': StrikeoutModelConfig.RANDOM_STATE,
        # Tunable parameters
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0), # Alias: colsample_bytree
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0), # Alias: subsample
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), # L2 regularization
    }

    fold_rmses = []
    n_cv_splits = 4 # Number of time series splits
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)

    logger.debug(f"Trial {trial.number}: Starting TimeSeriesSplit CV with {n_cv_splits} splits...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_data)):
        X_train_fold, X_val_fold = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train_fold, y_val_fold = y_data.iloc[train_idx], y_data.iloc[val_idx]
        if len(X_val_fold) == 0: continue

        train_set = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_set = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_set)

        callbacks = [lgb.early_stopping(100, verbose=False)] # Early stopping within fold

        model = lgb.train(lgb_params, train_set,
                          num_boost_round=1500, # High number, rely on early stopping
                          valid_sets=[train_set, val_set],
                          callbacks=callbacks)

        preds = model.predict(X_val_fold, num_iteration=model.best_iteration)
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
    parser = argparse.ArgumentParser(description="Train LightGBM Strikeout Model")
    parser.add_argument("--optuna-trials", type=int, default=StrikeoutModelConfig.OPTUNA_TRIALS, help="Number of Optuna trials.")
    parser.add_argument("--optuna-timeout", type=int, default=StrikeoutModelConfig.OPTUNA_TIMEOUT, help="Optuna timeout in seconds.")
    parser.add_argument("--use-best-params", action="store_true",
                        help="Automatically load the latest saved best parameters (skips Optuna).")
    parser.add_argument("--production", action="store_true", help="Train final model on all data (no Optuna).")
    parser.add_argument("--final-estimators", type=int, default=StrikeoutModelConfig.FINAL_ESTIMATORS, help="Max estimators for final model.")
    parser.add_argument("--early-stopping", type=int, default=StrikeoutModelConfig.EARLY_STOPPING_ROUNDS, help="Early stopping rounds for final model.")
    parser.add_argument("--verbose-fit", action="store_true", help="Enable verbose fitting output.")
    parser.add_argument("--top-n-features", type=int, default=None,
                        help="If set, select and save only the top N features based on importance AFTER training.")
    return parser.parse_args()

# --- Main Training Function ---
def train_model(args):
    """Loads data, trains LightGBM model (with optional Optuna), evaluates, and saves."""
    logger.info(f"Starting LightGBM training (Production Mode: {args.production})...")
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

    # --- Feature Selection (Initial Exclusion) ---
    logger.info("Selecting features using shared function...")
    # Use the full dataset `all_data` to get the potential features first
    potential_feature_cols, _ = select_features(
        all_data.copy(), # Pass a copy to avoid modification issues
        target_variable=StrikeoutModelConfig.TARGET_VARIABLE
    )
    if not potential_feature_cols:
        logger.error("No potential features selected after initial exclusion. Exiting.")
        sys.exit(1)
    logger.info(f"Found {len(potential_feature_cols)} potential features after initial exclusion.")

    # --- CORRECTED FEATURE LIST LOADING ---
    model_prefix = f"prod_{MODEL_TYPE}" if args.production else f"test_{MODEL_TYPE}"
    training_features = None # Initialize

    logger.info("Searching for the latest saved feature list file...")
    # Pattern to find *any* relevant feature list (full or top-N)
    any_list_pattern = f"{model_prefix}_*_feature_columns_*.pkl"
    latest_feature_file = find_latest_file(model_dir, any_list_pattern)

    if latest_feature_file:
        logger.info(f"Loading feature list from latest file: {latest_feature_file}")
        try:
            with open(latest_feature_file, 'rb') as f:
                training_features = pickle.load(f)
            logger.info(f"Loaded {len(training_features)} features from {latest_feature_file.name}")
        except Exception as e:
            logger.error(f"Failed to load feature list from {latest_feature_file}: {e}. Falling back to initial selection.")
            training_features = None # Reset on failure
    else:
        logger.warning("No saved feature list (.pkl) found.")

    # If loading failed or no file found, use the initially selected features
    if training_features is None:
        logger.info("Using initially selected features (fallback).")
        training_features = potential_feature_cols

    # Final check: Ensure loaded/selected features exist in the data
    missing_in_data = [f for f in training_features if f not in all_data.columns]
    if missing_in_data:
        logger.error(f"Features from loaded list are missing from data: {missing_in_data}.")
        # Filter out missing features from the list
        training_features = [f for f in training_features if f in all_data.columns]
        if not training_features:
             logger.error("No valid features remain after checking data columns. Exiting.")
             sys.exit(1)
        logger.warning(f"Proceeding with {len(training_features)} available features.")

    logger.info(f"Using {len(training_features)} features for this training run.")
    # --- END CORRECTED FEATURE LIST LOADING ---


    # --- Prepare Data Splits ---
    train_indices = all_data[all_data['season'].isin(StrikeoutModelConfig.DEFAULT_TRAIN_YEARS)].index
    test_indices = all_data[all_data['season'].isin(StrikeoutModelConfig.DEFAULT_TEST_YEARS)].index

    # Use the `training_features` list determined above
    X_train_full = all_data.loc[train_indices, training_features].copy()
    y_train_full = all_data.loc[train_indices, StrikeoutModelConfig.TARGET_VARIABLE].copy()
    X_test = all_data.loc[test_indices, training_features].copy()
    y_test = all_data.loc[test_indices, StrikeoutModelConfig.TARGET_VARIABLE].copy()

    # --- Handle Infinities/NaNs ---
    logger.info("Checking for infinite/NaN values before training...")
    for df_name, df_check in [('X_train_full', X_train_full), ('X_test', X_test)]:
        if not df_check.empty:
            inf_mask = np.isinf(df_check.select_dtypes(include=np.number)).any().any()
            nan_mask = df_check.isnull().any().any()
            if inf_mask or nan_mask:
                 logger.warning(f"Infinite or NaN values found in {df_name}. Replacing Inf->NaN, then imputing NaNs with train median.")
                 df_check.replace([np.inf, -np.inf], np.nan, inplace=True)
                 for col in df_check.columns[df_check.isnull().any()]:
                      if pd.api.types.is_numeric_dtype(df_check[col]): # Only impute numeric
                          median_val = X_train_full[col].median() # Use train median for consistency
                          fill_val = median_val if pd.notna(median_val) else 0
                          df_check[col].fillna(fill_val, inplace=True)

    logger.info(f"Train shape: {X_train_full.shape}, Test shape: {X_test.shape}")
    del all_data; gc.collect()

    # --- Hyperparameter Loading / Optuna ---
    best_params = None
    should_run_optuna = False
    param_pattern = f"{model_prefix}_best_params_*.json"

    if args.production:
        logger.info("--- Running in PRODUCTION MODE ---")
        logger.info("Attempting to load latest best parameters for production...")
        should_run_optuna = False
        latest_params_file = find_latest_file(model_dir, param_pattern)
        if latest_params_file:
            logger.info(f"Loading hyperparameters from: {latest_params_file}")
            try:
                with open(latest_params_file, 'r') as f: best_params = json.load(f)
                logger.info(f"Successfully loaded parameters for production.")
            except Exception as e: logger.error(f"Failed load prod params: {e}. Using defaults."); best_params = {}
        else: logger.warning(f"No param file found for production. Using defaults."); best_params = {}

    elif args.use_best_params:
        logger.info("Attempting load latest best params (--use-best-params specified)...")
        should_run_optuna = False
        latest_params_file = find_latest_file(model_dir, param_pattern)
        if latest_params_file:
            logger.info(f"Loading hyperparameters from: {latest_params_file}")
            try:
                with open(latest_params_file, 'r') as f: best_params = json.load(f)
                logger.info(f"Successfully loaded parameters.")
            except Exception as e: logger.error(f"Failed load params: {e}. Run Optuna."); best_params=None; should_run_optuna=True
        else: logger.warning(f"No param file found. Run Optuna."); best_params=None; should_run_optuna=True
    else:
        logger.info("No parameters loaded or specified (--use-best-params not set). Running Optuna.")
        should_run_optuna = True; best_params = None

    # Run Optuna if needed
    if should_run_optuna:
        logger.info("Running Optuna hyperparameter search...")
        optuna_start_time = time.time()
        study = optuna.create_study(direction='minimize') # Minimize RMSE
        try:
            # Pass only the selected features for tuning
            study.optimize(lambda trial: objective_lgbm_timeseries_rmse(trial, X_train_full, y_train_full),
                           n_trials=args.optuna_trials, timeout=args.optuna_timeout)
            best_params = study.best_params
            logger.info(f"Optuna finished in {(time.time() - optuna_start_time):.2f}s. Best Avg CV RMSE: {study.best_value:.4f}")
            logger.info(f"Best hyperparameters: {best_params}")
            # Save the new best parameters
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            params_save_path = model_dir / f"{model_prefix}_best_params_{timestamp}.json"
            try:
                with open(params_save_path, 'w') as f: json.dump(best_params, f, indent=4)
                logger.info(f"Saved best hyperparameters to: {params_save_path}")
            except Exception as e: logger.error(f"Failed to save best params: {e}")
        except Exception as e:
             logger.error(f"Optuna optimization failed: {e}", exc_info=True)
             logger.warning("Proceeding with default LightGBM parameters.")
             best_params = {} # Fallback

    # Ensure defaults if needed
    if best_params is None: best_params = {}
    if not best_params: # If empty dict after all attempts
        logger.warning("No valid hyperparameters. Using default parameters.")
        best_params = {'objective': 'regression_l1', 'metric': 'rmse', 'boosting_type': 'gbdt', 'n_jobs': -1, 'verbose': -1, 'seed': StrikeoutModelConfig.RANDOM_STATE, 'learning_rate': 0.05, 'num_leaves': 31} # Example defaults
    # Ensure core params exist
    best_params.setdefault('objective', 'regression_l1')
    best_params.setdefault('metric', 'rmse')
    best_params.setdefault('seed', StrikeoutModelConfig.RANDOM_STATE)
    best_params.setdefault('n_jobs', -1)
    best_params.setdefault('verbose', -1)

    # --- Final Model Training ---
    logger.info("Training final model on full training set (using early stopping)...")
    train_dataset = lgb.Dataset(X_train_full, label=y_train_full)
    valid_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)
    evals_result = {}
    callbacks = [
        lgb.early_stopping(args.early_stopping, verbose=args.verbose_fit),
        lgb.log_evaluation(period=100 if args.verbose_fit else 0) # Log every 100 rounds if verbose
    ]

    try:
        final_model = lgb.train(
            best_params,
            train_dataset,
            num_boost_round=args.final_estimators,
            valid_sets=[train_dataset, valid_dataset],
            valid_names=['train', 'eval'],
            callbacks=callbacks
        )
        logger.info("Final model training complete.")
        logger.info(f"Best iteration: {final_model.best_iteration}")
    except Exception as e:
        logger.error(f"Failed to train final model: {e}", exc_info=True); sys.exit(1)

    # --- Evaluation ---
    logger.info("Evaluating model...")
    train_preds = final_model.predict(X_train_full, num_iteration=final_model.best_iteration)
    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))
    train_mae = mean_absolute_error(y_train_full, train_preds)

    logger.info("--- Train Metrics ---")
    logger.info(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")

    if not X_test.empty and not y_test.empty:
        test_preds = final_model.predict(X_test, num_iteration=final_model.best_iteration)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_w1 = within_n_strikeouts(y_test, test_preds, n=1)
        test_w2 = within_n_strikeouts(y_test, test_preds, n=2)
        logger.info("--- Test Metrics ---")
        logger.info(f"Test RMSE : {test_rmse:.4f}, Test MAE : {test_mae:.4f}")
        logger.info(f"Test W/1 K: {test_w1:.4f}, Test W/2 K: {test_w2:.4f}")
    else:
        logger.info("Test set not available for evaluation.")

    # --- Save Artifacts ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_prefix already defined

    # Feature Importance
    try:
        importance_df = pd.DataFrame({
            'feature': final_model.feature_name(),
            'importance': final_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        imp_path = model_dir / f"{model_prefix}_feature_importance_full_{timestamp}.csv"
        importance_df.to_csv(imp_path, index=False)
        logger.info(f"Full importance list saved: {imp_path}")
        logger.info("Top 20 Feature Importances:")
        logger.info("\n" + importance_df.head(20).to_string(index=False))
    except Exception as e:
        logger.error(f"Could not get/save feature importances: {e}")

    # Importance Plot
    try:
        plt.figure(figsize=(10, max(8, len(training_features)//5))) # Adjust height based on features
        sns.barplot(x="importance", y="feature", data=importance_df.head(min(30, len(training_features)))) # Plot top 30 or fewer
        plt.title(f"LGBM Feature Importance ({model_prefix.replace('_lgb','')})")
        plt.tight_layout()
        plot_path = plot_dir / f"{model_prefix}_feat_imp_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"Importance plot: {plot_path}")
        plt.close()
    except Exception as e: logger.error(f"Failed to create importance plot: {e}")

    # Prediction vs Actual Plot
    try:
        plt.figure(figsize=(8, 8))
        if not X_test.empty: plt.scatter(y_test, test_preds, alpha=0.3, label='Test Set')
        if not X_train_full.empty: plt.scatter(y_train_full, train_preds, alpha=0.05, label='Train Set', color='orange')
        min_val = min(y_test.min() if not y_test.empty else 0, y_train_full.min() if not y_train_full.empty else 0)
        max_val = max(y_test.max() if not y_test.empty else 0, y_train_full.max() if not y_train_full.empty else 0)
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label='Perfect Prediction')
        plt.xlabel("Actual Strikeouts"); plt.ylabel("Predicted Strikeouts")
        plt.title(f"Actual vs. Predicted Strikeouts (LGBM - {model_prefix.replace('_lgb','')})")
        plt.legend(); plt.grid(True)
        plot_path = plot_dir / f"{model_prefix}_pred_actual_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"Pred/Actual plot: {plot_path}")
        plt.close()
    except Exception as e: logger.error(f"Failed to create pred/actual plot: {e}")

    # --- Save Top N Feature List (if --top-n-features N is specified for *this* run) ---
    # This happens AFTER training and evaluation
    if args.top_n_features is not None and args.top_n_features > 0:
        if 'importance_df' in locals() and not importance_df.empty: # Check if importance was calculated
             if args.top_n_features < len(importance_df):
                 top_n_features_list = importance_df['feature'].head(args.top_n_features).tolist()
                 logger.info(f"Selecting top {args.top_n_features} features based on importance from this run.")
                 top_n_features_path = model_dir / f"{model_prefix}_top_{args.top_n_features}_feature_columns_{timestamp}.pkl"
                 try:
                     with open(top_n_features_path, 'wb') as f: pickle.dump(top_n_features_list, f)
                     logger.info(f"Top {args.top_n_features} features list saved: {top_n_features_path}")
                 except Exception as e: logger.error(f"Failed to save top N feature list: {e}")
             else:
                 logger.warning(f"--top-n-features ({args.top_n_features}) >= total features ({len(importance_df)}). Saving full list used instead.")
                 # Save the full list used in this run (handled below)
        else:
             logger.warning("Importance DataFrame not available, cannot save top-N features list.")
             # Save the full list used in this run (handled below)

    # --- Save the FULL feature list used in THIS run ---
    # This is important for reproducibility, regardless of top-n flag for this run
    features_path = model_dir / f"{model_prefix}_feature_columns_{timestamp}.pkl"
    try:
        with open(features_path, 'wb') as f: pickle.dump(training_features, f) # Save the list actually used
        logger.info(f"Features list used for this run ({len(training_features)} features) saved: {features_path}")
    except Exception as e:
         logger.error(f"Failed to save feature list: {e}")

    # --- Save Model ---
    model_path = model_dir / f"{model_prefix}_strikeout_model_{timestamp}.txt"
    try:
        final_model.save_model(str(model_path))
        logger.info(f"Model saved: {model_path}")
    except Exception as e: logger.error(f"Failed to save model: {e}")

    # Save best params (if found via Optuna in this run)
    if should_run_optuna and best_params and 'best_cv_score_from_optuna' not in best_params: # Avoid resaving loaded params
         params_save_path_final = model_dir / f"{model_prefix}_best_params_{timestamp}.json"
         try:
             params_to_save = best_params.copy()
             params_to_save['best_iteration_from_final_train'] = final_model.best_iteration
             params_to_save['best_cv_score_from_optuna'] = study.best_value
             with open(params_save_path_final, 'w') as f: json.dump(params_to_save, f, indent=4)
             logger.info(f"Saved final hyperparameters (from Optuna) to: {params_save_path_final}")
         except Exception as e: logger.error(f"Failed to save Optuna best parameters: {e}")
    elif best_params:
         logger.info(f"Final model trained using loaded parameters.")

    logger.info("Model training finished successfully.")


# --- Main Execution Block ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    args = parse_args()
    train_model(args)
    logger.info("--- LightGBM Training Script Completed ---")





