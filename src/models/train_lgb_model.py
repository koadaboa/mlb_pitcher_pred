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
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_poisson_deviance
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# --- Imports and Setup ---
try:
    # Import specific config class directly
    from src.config import DBConfig, StrikeoutModelConfig, LogConfig, FileConfig
    from src.data.utils import setup_logger, DBConnection, find_latest_file
    from src.features.selection import select_features
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    MODULE_IMPORTS_OK = False
    sys.exit(1)

# Setup logger
LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger('train_lgb_model', LogConfig.LOG_DIR / 'train_lgb_model.log')

MODEL_TYPE = 'lgb'

# --- Evaluation Metric Helper --- (Keep as is)
def within_n_strikeouts(y_true, y_pred, n=1):
    """ Calculates percentage of predictions within n strikeouts """
    if y_true is None or y_pred is None or len(y_true) != len(y_pred): return np.nan
    y_true_arr = np.asarray(y_true); y_pred_arr = np.asarray(y_pred)
    return np.mean(np.abs(y_true_arr - np.round(y_pred_arr)) <= n)

# --- Optuna Objective Function (MODIFIED to use config) ---
def objective_lgbm_timeseries_poisson(trial, X_data, y_data):
    """ Optuna objective function minimizing Poisson deviance via CV """
    # Base parameters from config
    lgb_params = StrikeoutModelConfig.LGBM_BASE_PARAMS.copy() # Start with base

    # Define search space using ranges from config
    grid = StrikeoutModelConfig.LGBM_PARAM_GRID
    lgb_params.update({
        'learning_rate': trial.suggest_float('learning_rate', grid['learning_rate'][0], grid['learning_rate'][1], log=True),
        'num_leaves': trial.suggest_int('num_leaves', grid['num_leaves'][0], grid['num_leaves'][1]),
        'max_depth': trial.suggest_int('max_depth', grid['max_depth'][0], grid['max_depth'][1]),
        'min_child_samples': trial.suggest_int('min_child_samples', grid['min_child_samples'][0], grid['min_child_samples'][1]),
        'feature_fraction': trial.suggest_float('feature_fraction', grid['feature_fraction'][0], grid['feature_fraction'][1]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', grid['bagging_fraction'][0], grid['bagging_fraction'][1]),
        'bagging_freq': trial.suggest_int('bagging_freq', grid['bagging_freq'][0], grid['bagging_freq'][1]),
        'reg_alpha': trial.suggest_float('reg_alpha', grid['reg_alpha'][0], grid['reg_alpha'][1], log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', grid['reg_lambda'][0], grid['reg_lambda'][1], log=True),
    })

    fold_devs = []
    # Use CV splits from config
    tscv = TimeSeriesSplit(n_splits=StrikeoutModelConfig.OPTUNA_CV_SPLITS)

    logger.debug(f"Trial {trial.number}: Starting TimeSeriesSplit CV with {tscv.n_splits} splits...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_data)):
        X_train_fold, X_val_fold = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train_fold, y_val_fold = y_data.iloc[train_idx], y_data.iloc[val_idx]
        if len(X_val_fold) == 0: continue

        train_set = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_set = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_set)

        # Use early stopping rounds from config for Optuna CV folds as well
        callbacks = [lgb.early_stopping(StrikeoutModelConfig.EARLY_STOPPING_ROUNDS // 2, verbose=False)] # Use half for CV

        model = lgb.train(lgb_params, train_set,
                          num_boost_round=StrikeoutModelConfig.FINAL_ESTIMATORS, # Use max estimators from config
                          valid_sets=[train_set, val_set],
                          callbacks=callbacks)

        preds = model.predict(X_val_fold, num_iteration=model.best_iteration)
        dev = mean_poisson_deviance(y_val_fold, preds)
        fold_devs.append(dev)
        logger.debug(f"  Trial {trial.number} Fold {fold+1}/{tscv.n_splits} - Val Poisson Deviance: {dev:.4f}")
        # Pruning (optional): report intermediate value and check if trial should be pruned
        trial.report(dev, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if not fold_devs:
        return float('inf') # Should not happen if data is valid
    return float(np.mean(fold_devs))

# --- Argument Parser (SIMPLIFIED) ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train LightGBM Strikeout Model")
    # Simplified Mode Selection
    parser.add_argument("--mode", type=str, default="tune", choices=['tune', 'retrain', 'production'],
                        help="Training mode: 'tune' (run Optuna), 'retrain' (load best params, train on train set), 'production' (load best params, train on all data - define carefully!). Default: tune")
    # Simplified Feature Selection
    parser.add_argument("--feature-selection", type=str, default="vif_shap", choices=['none', 'vif', 'shap', 'vif_shap'],
                        help="Feature selection method to apply before tuning/training. Default: vif_shap")
    # Optional override for Optuna trials
    parser.add_argument("--optuna-trials", type=int, default=None,
                        help=f"Override number of Optuna trials (default from config: {StrikeoutModelConfig.OPTUNA_TRIALS}). Only used if mode='tune'.")
    # Optional override for saving top N features
    parser.add_argument("--top-n-features", type=int, default=None,
                        help="Save a list of the top N features based on importance AFTER training.")
    # Flag to force production artifacts (different naming)
    parser.add_argument("--prod-artifact", action="store_true",
                        help="Save artifacts with 'prod_' prefix instead of 'test_'.")
    # Flag for more verbose logging during LGBM training
    parser.add_argument("--verbose-fit", action="store_true",
                        help="Enable verbose fitting output during final LGBM training.")

    return parser.parse_args()

# --- Main Training Function (MODIFIED to use config and simplified args) ---
def train_model(args):
    """Loads data, trains LightGBM model (with optional Optuna), evaluates, and saves."""
    logger.info(f"Starting LightGBM training (Mode: {args.mode}, Feature Selection: {args.feature_selection})...")
    db_path = Path(DBConfig.PATH)
    model_dir = Path(FileConfig.MODELS_DIR)
    plot_dir = Path(FileConfig.PLOTS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = f"prod_{MODEL_TYPE}" if args.prod_artifact else f"test_{MODEL_TYPE}"
    logger.info(f"Artifact prefix: {model_prefix}")

    # --- Load Data (Update table names) ---
    logger.info("Loading feature data from basic tables...")
    try:
        with DBConnection(db_path) as conn:
            # Load from the new basic feature tables
            train_table = "historical_features_basic_train"
            test_table = "historical_features_basic_test"
            train_df = pd.read_sql_query(f"SELECT * FROM {train_table}", conn)
            logger.info(f"Loaded {len(train_df)} train rows from '{train_table}'")
            test_df = pd.read_sql_query(f"SELECT * FROM {test_table}", conn)
            logger.info(f"Loaded {len(test_df)} test rows from '{test_table}'")
        # Combine for initial feature selection if needed, otherwise keep separate
        # Let's assume select_features works on train_df only for now
        # all_data = pd.concat([train_df, test_df], ignore_index=True)
        # logger.info(f"Total rows loaded: {len(all_data)}")
        # logger.info(f"Available columns ({len(train_df.columns)}): {train_df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error loading feature data: {e}", exc_info=True); sys.exit(1)

    # --- Feature Selection ---
    logger.info(f"Applying feature selection method: {args.feature_selection}")
    # Apply selection primarily to training data columns
    train_cols = train_df.columns.tolist()
    # *** MODIFIED CALL: Removed cols_to_keep_explicit ***
    potential_feature_cols, _ = select_features(
        train_df.copy(), # Pass train_df for selection logic
        target_variable=StrikeoutModelConfig.TARGET_VARIABLE,
        prune_vif=(args.feature_selection in ['vif', 'vif_shap']), # Enable based on flag
        vif_threshold=StrikeoutModelConfig.VIF_THRESHOLD, # Use config value
        prune_shap=(args.feature_selection in ['shap', 'vif_shap']), # Enable based on flag
        shap_model=None, # Requires loading model BEFORE selection - adjust if needed
        shap_threshold=StrikeoutModelConfig.SHAP_THRESHOLD, # Use config value
        shap_sample_frac=StrikeoutModelConfig.SHAP_SAMPLE_FRAC # Use config value
        # cols_to_keep_explicit = [] # <<< REMOVED THIS LINE >>>
    )
    if not potential_feature_cols:
        logger.error("No potential features selected after initial exclusion. Exiting.")
        sys.exit(1)

    # Use the selected features
    training_features = potential_feature_cols
    logger.info(f"Using {len(training_features)} features after selection: {training_features}")

    # Ensure target variable is not accidentally included in features
    if StrikeoutModelConfig.TARGET_VARIABLE in training_features:
        logger.warning(f"Target variable '{StrikeoutModelConfig.TARGET_VARIABLE}' found in feature list. Removing.")
        training_features.remove(StrikeoutModelConfig.TARGET_VARIABLE)

    # --- Prepare Data Splits using selected features ---
    X_train_full = train_df[training_features].copy()
    y_train_full = train_df[StrikeoutModelConfig.TARGET_VARIABLE].copy()
    # Ensure test set uses the same features
    X_test = test_df[[f for f in training_features if f in test_df.columns]].copy()
    y_test = test_df[StrikeoutModelConfig.TARGET_VARIABLE].copy()

    # Add check for missing columns in test set AFTER selection
    missing_test_cols = [f for f in training_features if f not in X_test.columns]
    if missing_test_cols:
        logger.warning(f"Columns selected from train set missing in test set: {missing_test_cols}. Adding as NaN.")
        for col in missing_test_cols:
            X_test[col] = np.nan


    # --- Handle Infinities/NaNs --- (Keep this logic)
    logger.info("Checking for infinite/NaN values before training...")
    # ... (NaN/Inf handling code remains the same) ...
    for df_name, df_check in [('X_train_full', X_train_full), ('X_test', X_test)]:
        if not df_check.empty:
            inf_mask = np.isinf(df_check.select_dtypes(include=np.number)).any().any()
            nan_mask = df_check.isnull().any().any()
            if inf_mask or nan_mask:
                 logger.warning(f"Infinite or NaN values found in {df_name}. Replacing Inf->NaN, then imputing NaNs with train median/0.")
                 df_check.replace([np.inf, -np.inf], np.nan, inplace=True) # Keep non-deprecated replace
                 for col in df_check.columns[df_check.isnull().any()]:
                      if pd.api.types.is_numeric_dtype(df_check[col]):
                          median_val = X_train_full[col].median()
                          fill_val = median_val if pd.notna(median_val) else 0
                          # Use assignment instead of inplace=True
                          df_check[col] = df_check[col].fillna(fill_val)


    logger.info(f"Train shape: {X_train_full.shape}, Test shape: {X_test.shape}")
    del train_df, test_df; gc.collect() # Clear original loaded data

    # --- Hyperparameter Loading / Optuna ---
    best_params = None
    param_pattern = f"{model_prefix}_best_params_*.json"
    should_run_optuna = (args.mode == 'tune')

    if not should_run_optuna:
        # Load params if mode is 'retrain' or 'production'
        logger.info(f"Mode '{args.mode}': Attempting to load latest parameters...")
        latest_params_file = find_latest_file(model_dir, param_pattern)
        if latest_params_file:
            logger.info(f"Loading hyperparameters from: {latest_params_file}")
            try:
                with open(latest_params_file, 'r') as f: best_params = json.load(f)
                # Remove optuna/training specific keys if they exist from previous save
                best_params.pop('best_iteration_from_final_train', None)
                best_params.pop('best_cv_score_from_optuna', None)
                logger.info(f"Successfully loaded parameters.")
            except Exception as e:
                logger.error(f"Failed load params: {e}. Using defaults from config.")
                best_params = StrikeoutModelConfig.LGBM_BASE_PARAMS.copy() # Fallback to base
        else:
            logger.warning(f"No param file found matching {param_pattern}. Using defaults from config.")
            best_params = StrikeoutModelConfig.LGBM_BASE_PARAMS.copy() # Fallback to base

    else: # Mode is 'tune'
        logger.info("Running Optuna hyperparameter search...")
        optuna_start_time = time.time()
        # Use trials from args if provided, otherwise from config
        n_trials = args.optuna_trials if args.optuna_trials is not None else StrikeoutModelConfig.OPTUNA_TRIALS
        study = optuna.create_study(direction='minimize')
        try:
            study.optimize(lambda trial: objective_lgbm_timeseries_poisson(trial, X_train_full, y_train_full),
                           n_trials=n_trials,
                           timeout=StrikeoutModelConfig.OPTUNA_TIMEOUT) # Use timeout from config
            best_params = study.best_params
            # Combine with base params for final training
            final_params = StrikeoutModelConfig.LGBM_BASE_PARAMS.copy()
            final_params.update(best_params) # Update base with tuned params
            best_params = final_params # Use the combined dict going forward
            logger.info(f"Optuna finished in {(time.time() - optuna_start_time):.2f}s. Best Avg CV Poisson Deviance: {study.best_value:.4f}")
            logger.info(f"Best hyperparameters found: {best_params}")
            # Save the new best parameters (including base ones)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            params_save_path = model_dir / f"{model_prefix}_best_params_{timestamp}.json"
            try:
                params_to_save = best_params.copy()
                params_to_save['best_cv_score_from_optuna'] = study.best_value # Add score
                with open(params_save_path, 'w') as f: json.dump(params_to_save, f, indent=4)
                logger.info(f"Saved best hyperparameters to: {params_save_path}")
            except Exception as e: logger.error(f"Failed to save best params: {e}")
        except Exception as e:
             logger.error(f"Optuna optimization failed: {e}", exc_info=True)
             logger.warning("Proceeding with default LightGBM parameters from config.")
             best_params = StrikeoutModelConfig.LGBM_BASE_PARAMS.copy() # Fallback


    # --- Final Model Training ---
    logger.info("Preparing data for final model training...")
    # Use appropriate data based on mode
    if args.mode == 'production':
        logger.warning("Production mode: Training on ALL available data (Train + Test). Ensure this is intended.")
        X_final_train = pd.concat([X_train_full, X_test], ignore_index=True)
        y_final_train = pd.concat([y_train_full, y_test], ignore_index=True)
        train_dataset = lgb.Dataset(X_final_train, label=y_final_train)
        valid_sets = [train_dataset] # No validation set when training on all data
        valid_names = ['train']
        early_stopping_rounds = None # Disable early stopping when training on all data
        logger.info(f"Final training on {len(X_final_train)} rows.")
    else: # 'tune' or 'retrain' mode - use train set, validate on test set
        X_final_train = X_train_full
        y_final_train = y_train_full
        train_dataset = lgb.Dataset(X_final_train, label=y_final_train)
        valid_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)
        valid_sets = [train_dataset, valid_dataset]
        valid_names = ['train', 'eval']
        early_stopping_rounds = StrikeoutModelConfig.EARLY_STOPPING_ROUNDS # Use config value
        logger.info(f"Final training on {len(X_final_train)} rows, validating on {len(X_test)} rows.")

    # Set verbosity for final fit
    verbose_eval = StrikeoutModelConfig.VERBOSE_FIT_FREQUENCY if args.verbose_fit else 0
    callbacks = []
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
         callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=args.verbose_fit))
    callbacks.append(lgb.log_evaluation(period=verbose_eval))

    logger.info("Training final model...")
    try:
        final_model = lgb.train(
            best_params, # Use loaded or tuned params
            train_dataset,
            num_boost_round=StrikeoutModelConfig.FINAL_ESTIMATORS, # Max estimators from config
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        best_iter = final_model.best_iteration if early_stopping_rounds else StrikeoutModelConfig.FINAL_ESTIMATORS
        logger.info(f"Final model training complete. Best iteration: {best_iter}")
    except Exception as e:
        logger.error(f"Failed to train final model: {e}", exc_info=True); sys.exit(1)

    # --- Evaluation ---
    logger.info("Evaluating model...")
    train_preds = final_model.predict(X_final_train, num_iteration=best_iter) # Use X_final_train here
    train_rmse = np.sqrt(mean_squared_error(y_final_train, train_preds))
    train_mae = mean_absolute_error(y_final_train, train_preds)
    train_pdev = mean_poisson_deviance(y_final_train, train_preds)
    # Calculate train Within N K if needed (usually focus on test)
    # train_w1 = within_n_strikeouts(y_final_train, train_preds, n=1)
    # train_w2 = within_n_strikeouts(y_final_train, train_preds, n=2)

    logger.info("--- Train Metrics ---")
    logger.info(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
    logger.info(f"Train Poisson Deviance: {train_pdev:.4f}")
    # logger.info(f"Train W/1 K: {train_w1:.4f}, Train W/2 K: {train_w2:.4f}") # Optional

    if not X_test.empty and not y_test.empty:
        test_preds = final_model.predict(X_test, num_iteration=best_iter)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_w1 = within_n_strikeouts(y_test, test_preds, n=1)
        test_w2 = within_n_strikeouts(y_test, test_preds, n=2)
        test_pdev = mean_poisson_deviance(y_test, test_preds)

        logger.info("--- Test Metrics ---")
        logger.info(f"Test RMSE : {test_rmse:.4f}, Test MAE : {test_mae:.4f}")
        logger.info(f"Test W/1 K: {test_w1:.4f}, Test W/2 K: {test_w2:.4f}")
        logger.info(f"Test Poisson Deviance: {test_pdev:.4f}")
    else:
        logger.info("Test set not available for evaluation.")

    # --- Save Artifacts ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_prefix already defined

    # Feature Importance Plot (Keep as is)
    try:
        # ... (importance calculation and plotting) ...
        importance_df = pd.DataFrame({
            'feature': final_model.feature_name(),
            'importance': final_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        # ... (save csv) ...
        # ... (generate plot) ...
        plt.figure(figsize=(10, max(8, len(training_features)//5)))
        sns.barplot(x="importance", y="feature", data=importance_df.head(min(30, len(training_features)))) # Plot top 30 or fewer
        plt.title(f"LGBM Feature Importance ({model_prefix.replace('_lgb','')})")
        plt.tight_layout()
        plot_path = plot_dir / f"{model_prefix}_feat_imp_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"Importance plot: {plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Could not get/save/plot feature importances: {e}")


    # Prediction vs Actual Plot (Keep as is)
    try:
        # ... (generate plot) ...
        plt.figure(figsize=(8, 8))
        if not X_test.empty: plt.scatter(y_test, test_preds, alpha=0.3, label='Test Set')
        # Use train data corresponding to X_final_train for plotting if needed
        plt.scatter(y_final_train, train_preds, alpha=0.05, label='Train Set (Final)', color='orange')
        min_val = min(y_test.min() if not y_test.empty else 0, y_final_train.min() if not y_final_train.empty else 0)
        max_val = max(y_test.max() if not y_test.empty else 0, y_final_train.max() if not y_final_train.empty else 0)
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label='Perfect Prediction')
        plt.xlabel("Actual Strikeouts"); plt.ylabel("Predicted Strikeouts")
        plt.title(f"Actual vs. Predicted Strikeouts (LGBM - {model_prefix.replace('_lgb','')})")
        plt.legend(); plt.grid(True)
        plot_path = plot_dir / f"{model_prefix}_pred_actual_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"Pred/Actual plot: {plot_path}")
        plt.close()
    except Exception as e: logger.error(f"Failed to create pred/actual plot: {e}")

    # --- Save Top N Feature List (Keep as is) ---
    # ... (logic for saving top N based on args.top_n_features) ...

    # --- Save the FULL feature list used in THIS run --- (Keep as is)
    features_path = model_dir / f"{model_prefix}_feature_columns_{timestamp}.pkl"
    try:
        with open(features_path, 'wb') as f: pickle.dump(training_features, f) # Save the list actually used
        logger.info(f"Features list used ({len(training_features)}) saved: {features_path}")
    except Exception as e: logger.error(f"Failed to save feature list: {e}")

    # --- Save Model --- (Keep as is)
    model_path = model_dir / f"{model_prefix}_strikeout_model_{timestamp}.txt"
    try:
        final_model.save_model(str(model_path))
        logger.info(f"Model saved: {model_path}")
    except Exception as e: logger.error(f"Failed to save model: {e}")

    # --- Save params used for final train --- (Keep as is)
    params_used_path = model_dir / f"{model_prefix}_params_used_{timestamp}.json"
    try:
        # ... (logic to save best_params + best_iter + optuna score if applicable) ...
        params_to_save = best_params.copy()
        params_to_save['best_iteration_used'] = best_iter
        if should_run_optuna and 'study' in locals():
            params_to_save['best_cv_score_from_optuna'] = study.best_value
        with open(params_used_path, 'w') as f: json.dump(params_to_save, f, indent=4)
        logger.info(f"Parameters used for training saved: {params_used_path}")
    except Exception as e: logger.error(f"Failed to save parameters used: {e}")


    logger.info("Model training finished successfully.") # This line should now come AFTER metric logging


# --- Main Execution Block ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    args = parse_args()
    train_model(args)
    logger.info("--- LightGBM Training Script Completed ---")