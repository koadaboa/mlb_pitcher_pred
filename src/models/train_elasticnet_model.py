# src/models/train_elasticnet_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet # Import ElasticNet
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
import joblib # Use joblib for saving ElasticNet model and scaler
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
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Assuming script is run via python -m src.models.train_elasticnet_model
try:
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
logger = setup_logger('train_elasticnet_model', LogConfig.LOG_DIR / 'train_elasticnet_model.log')

MODEL_TYPE = 'enet' # Identifier for filenames

# --- Evaluation Metric Helper ---
def within_n_strikeouts(y_true, y_pred, n=1):
    """ Calculates percentage of predictions within n strikeouts """
    if y_true is None or y_pred is None or len(y_true) != len(y_pred): return np.nan
    y_true_arr = np.asarray(y_true); y_pred_arr = np.asarray(y_pred)
    return np.mean(np.abs(y_true_arr - np.round(y_pred_arr)) <= n)

# --- Optuna Objective Function (using TimeSeriesSplit for ElasticNet) ---
def objective_enet_timeseries_rmse(trial, X_data, y_data):
    """ Optuna objective function using TimeSeriesSplit CV for ElasticNet """
    # Define search space for ElasticNet
    # Alpha (overall regularization strength) - log scale often works well
    alpha = trial.suggest_float('alpha', 1e-4, 1e2, log=True)
    # L1 Ratio (mix between L1 and L2) - 0 is Ridge, 1 is Lasso
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

    enet_params = {
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'max_iter': 2000, # Increase max iterations for convergence
        'tol': 1e-3,      # Tolerance for stopping criteria
        'random_state': StrikeoutModelConfig.RANDOM_STATE,
        'selection': 'random' # Use random selection for potentially faster convergence on large datasets
    }

    fold_rmses = []
    n_cv_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    scaler = StandardScaler() # Need to scale within each fold

    logger.debug(f"Trial {trial.number}: Starting ElasticNet TimeSeriesSplit CV (alpha={alpha:.4f}, l1_ratio={l1_ratio:.4f})...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_data)):
        X_train_fold, X_val_fold = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train_fold, y_val_fold = y_data.iloc[train_idx], y_data.iloc[val_idx]
        if len(X_val_fold) == 0: continue

        # --- Scaling within CV fold ---
        try:
            X_train_fold_scaled = scaler.fit_transform(X_train_fold)
            X_val_fold_scaled = scaler.transform(X_val_fold)
        except Exception as e:
             logger.error(f"Trial {trial.number} Fold {fold+1}: Error scaling data: {e}")
             continue # Skip fold if scaling fails

        model = ElasticNet(**enet_params)

        try:
            model.fit(X_train_fold_scaled, y_train_fold)
            preds = model.predict(X_val_fold_scaled)
            # Ensure predictions are non-negative (strikeouts cannot be negative)
            preds = np.maximum(0, preds)
            rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
            fold_rmses.append(rmse)
            logger.debug(f"  Trial {trial.number} Fold {fold+1}/{n_cv_splits} - Val RMSE: {rmse:.4f}")
        except Exception as e:
             logger.error(f"Trial {trial.number} Fold {fold+1}: Error fitting/predicting ElasticNet: {e}")
             fold_rmses.append(float('inf')) # Penalize failed trials


    if not fold_rmses: return float('inf')
    average_rmse = np.mean(fold_rmses)
    logger.debug(f"Trial {trial.number} completed. Average CV RMSE: {average_rmse:.4f}")
    return average_rmse

# --- Argument Parser ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ElasticNet Strikeout Model")
    # ElasticNet is fast, so use default Optuna trials
    parser.add_argument("--optuna-trials", type=int, default=StrikeoutModelConfig.OPTUNA_TRIALS, help="Number of Optuna trials.")
    parser.add_argument("--optuna-timeout", type=int, default=StrikeoutModelConfig.OPTUNA_TIMEOUT, help="Optuna timeout in seconds.")
    parser.add_argument("--use-best-params", action="store_true",
                        help="Automatically load the latest saved best parameters for ElasticNet (skips Optuna).")
    parser.add_argument("--production", action="store_true", help="Train final model on all data (no Optuna).")
    # ElasticNet doesn't use n_estimators or early stopping
    # parser.add_argument("--verbose-fit", action="store_true", help="Enable verbose fitting output.") # ElasticNet doesn't have verbose fit
    parser.add_argument("--top-n-features", type=int, default=None,
                        help="Load top N features list (if specified and available). ElasticNet does not generate its own top-N list based on importance.")
    return parser.parse_args()

# --- Main Training Function ---
def train_model(args):
    """Loads data, trains ElasticNet model (with optional Optuna), evaluates, and saves."""
    logger.info(f"Starting ElasticNet training (Production Mode: {args.production})...")
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

    # --- Load Correct Feature List (Full or Top-N from other models) ---
    model_prefix = "prod_enet" if args.production else "test_enet" # Add enet identifier
    logger.info("Searching for the latest saved feature list file (could be from LGBM/XGB)...")
    # Prioritize loading a top-N list if specified and available
    feature_list_file_to_load = None
    if args.top_n_features and args.top_n_features > 0:
         top_n_pattern = f"*_top_{args.top_n_features}_feature_columns_*.pkl"
         latest_top_n_file = find_latest_file(model_dir, top_n_pattern)
         if latest_top_n_file:
              logger.info(f"Found a top-{args.top_n_features} feature list: {latest_top_n_file.name}")
              feature_list_file_to_load = latest_top_n_file
         else:
              logger.warning(f"Specified --top-n-features {args.top_n_features}, but no matching .pkl file found.")

    if not feature_list_file_to_load:
         logger.info("Searching for the latest *full* feature list file (could be from lgbm/xgb)...")
         any_full_list_pattern = "*_feature_columns_*.pkl"
         latest_full_feature_file = find_latest_file(model_dir, any_full_list_pattern)
         if latest_full_feature_file:
              logger.info(f"Found latest full feature list: {latest_full_feature_file.name}")
              feature_list_file_to_load = latest_full_feature_file

    if feature_list_file_to_load:
        logger.info(f"Loading feature list from: {feature_list_file_to_load}")
        try:
            with open(feature_list_file_to_load, 'rb') as f:
                training_features = pickle.load(f)
            logger.info(f"Loaded {len(training_features)} features from {feature_list_file_to_load.name}")
        except Exception as e:
            logger.error(f"Failed to load feature list from {feature_list_file_to_load}: {e}. Falling back.")
            training_features = potential_feature_cols
    else:
        logger.warning("No saved feature list (.pkl) found. Using initially selected features.")
        training_features = potential_feature_cols

    missing_in_data = [f for f in training_features if f not in all_data.columns]
    if missing_in_data:
        logger.error(f"Features in loaded list missing from data: {missing_in_data}.")
        training_features = [f for f in training_features if f in all_data.columns]
        if not training_features:
             logger.error("No valid features remain after checking data. Exiting.")
             sys.exit(1)
    logger.info(f"Using {len(training_features)} features for this ElasticNet training run.")

    # --- Prepare Data Splits ---
    train_indices = all_data[all_data['season'].isin(StrikeoutModelConfig.DEFAULT_TRAIN_YEARS)].index
    test_indices = all_data[all_data['season'].isin(StrikeoutModelConfig.DEFAULT_TEST_YEARS)].index

    X_train_full = all_data.loc[train_indices, training_features].copy()
    y_train_full = all_data.loc[train_indices, StrikeoutModelConfig.TARGET_VARIABLE].copy()
    X_test = all_data.loc[test_indices, training_features].copy()
    y_test = all_data.loc[test_indices, StrikeoutModelConfig.TARGET_VARIABLE].copy()

    # --- Handle Infinities/NaNs (BEFORE Scaling) ---
    logger.info("Checking for infinite/NaN values before scaling...")
    for df_name, df_check in [('X_train_full', X_train_full), ('X_test', X_test)]:
        if not df_check.empty:
            inf_mask = np.isinf(df_check).any().any()
            nan_mask = df_check.isnull().any().any()
            if inf_mask or nan_mask:
                 logger.warning(f"Infinite or NaN values found in {df_name}. Replacing Inf->NaN, then imputing NaNs with train median.")
                 df_check.replace([np.inf, -np.inf], np.nan, inplace=True)
                 for col in df_check.columns[df_check.isnull().any()]:
                      median_val = X_train_full[col].median()
                      fill_val = median_val if pd.notna(median_val) else 0
                      df_check[col].fillna(fill_val, inplace=True)

    logger.info(f"Train shape: {X_train_full.shape}, Test shape: {X_test.shape}")
    del all_data; gc.collect()

    # --- Feature Scaling ---
    logger.info("Scaling features using StandardScaler (fit on train only)...")
    scaler = StandardScaler()
    try:
        X_train_full_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test) if not X_test.empty else X_test
        logger.info("Feature scaling complete.")
    except Exception as e:
        logger.error(f"Error during feature scaling: {e}", exc_info=True); sys.exit(1)

    # --- Hyperparameter Loading / Optuna ---
    best_params = None
    should_run_optuna = False
    param_pattern = f"{model_prefix}_best_params_*.json" # Use specific prefix

    if args.production:
        logger.info("--- Running in PRODUCTION MODE (ElasticNet) ---")
        logger.info("Attempting to load latest best ElasticNet parameters for production...")
        should_run_optuna = False
        latest_params_file = find_latest_file(model_dir, param_pattern)
        if latest_params_file:
            logger.info(f"Loading hyperparameters from: {latest_params_file}")
            try:
                with open(latest_params_file, 'r') as f: best_params = json.load(f)
                logger.info(f"Successfully loaded ElasticNet parameters for production.")
            except Exception as e: logger.error(f"Failed load prod params: {e}. Using defaults."); best_params = {}
        else: logger.warning(f"No ElasticNet param file found for production. Using defaults."); best_params = {}

    elif args.use_best_params:
        logger.info("Attempting load latest best ElasticNet params (--use-best-params specified)...")
        should_run_optuna = False
        latest_params_file = find_latest_file(model_dir, param_pattern)
        if latest_params_file:
            logger.info(f"Loading hyperparameters from: {latest_params_file}")
            try:
                with open(latest_params_file, 'r') as f: best_params = json.load(f)
                logger.info(f"Successfully loaded ElasticNet parameters.")
            except Exception as e: logger.error(f"Failed load params: {e}. Run Optuna."); best_params=None; should_run_optuna=True
        else: logger.warning(f"No ElasticNet param file found. Run Optuna."); best_params=None; should_run_optuna=True
    else:
        logger.info("No parameters loaded or specified (--use-best-params not set). Running Optuna for ElasticNet.")
        should_run_optuna = True; best_params = None

    # Run Optuna if needed
    if should_run_optuna:
        logger.info("Running Optuna hyperparameter search for ElasticNet...")
        optuna_start_time = time.time()
        study = optuna.create_study(direction='minimize')
        try:
            # Pass UN-SCALED data to Optuna - scaling happens inside objective
            study.optimize(lambda trial: objective_enet_timeseries_rmse(trial, X_train_full, y_train_full),
                           n_trials=args.optuna_trials, timeout=args.optuna_timeout)
            best_params = study.best_params
            logger.info(f"Optuna finished in {(time.time() - optuna_start_time):.2f}s. Best Avg CV RMSE: {study.best_value:.4f}")
            logger.info(f"Best ElasticNet hyperparameters: {best_params}")
            # Save the new best parameters
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            params_save_path = model_dir / f"{model_prefix}_best_params_{timestamp}.json"
            try:
                with open(params_save_path, 'w') as f: json.dump(best_params, f, indent=4)
                logger.info(f"Saved best hyperparameters to: {params_save_path}")
            except Exception as e: logger.error(f"Failed to save best params: {e}")
        except Exception as e:
             logger.error(f"Optuna optimization failed: {e}", exc_info=True)
             logger.warning("Proceeding with default ElasticNet parameters.")
             best_params = {} # Fallback

    # Ensure defaults if needed
    if best_params is None: best_params = {}
    if not best_params: # If empty dict after all attempts
        logger.warning("No valid ElasticNet hyperparameters. Using default parameters.")
        best_params = {'alpha': 1.0, 'l1_ratio': 0.5} # Example defaults
    # Ensure core params exist
    best_params.setdefault('alpha', 1.0)
    best_params.setdefault('l1_ratio', 0.5)
    best_params.setdefault('random_state', StrikeoutModelConfig.RANDOM_STATE)
    best_params.setdefault('max_iter', 2000)
    best_params.setdefault('tol', 1e-3)


    # --- Final Model Training ---
    logger.info("Training final ElasticNet model on full SCALED training data...")
    final_model = ElasticNet(**best_params)

    try:
        final_model.fit(X_train_full_scaled, y_train_full)
        logger.info("Final ElasticNet model training complete.")
    except Exception as e:
        logger.error(f"Failed to train final ElasticNet model: {e}", exc_info=True); sys.exit(1)

    # --- Evaluation ---
    logger.info("Evaluating ElasticNet model...")
    # Predict on SCALED data
    train_preds = final_model.predict(X_train_full_scaled)
    train_preds = np.maximum(0, train_preds) # Ensure non-negative predictions
    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))
    train_mae = mean_absolute_error(y_train_full, train_preds)

    logger.info("--- Train Metrics (ElasticNet) ---")
    logger.info(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")

    if not X_test.empty and not y_test.empty:
        test_preds = final_model.predict(X_test_scaled) # Use scaled test data
        test_preds = np.maximum(0, test_preds) # Ensure non-negative predictions
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_w1 = within_n_strikeouts(y_test, test_preds, n=1)
        test_w2 = within_n_strikeouts(y_test, test_preds, n=2)
        logger.info("--- Test Metrics (ElasticNet) ---")
        logger.info(f"Test RMSE : {test_rmse:.4f}, Test MAE : {test_mae:.4f}")
        logger.info(f"Test W/1 K: {test_w1:.4f}, Test W/2 K: {test_w2:.4f}")
    else:
        logger.info("Test set not available for evaluation.")

    # --- Save Artifacts ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_prefix already defined

    # --- Feature Coefficients (as Importance) ---
    try:
        coefficients = final_model.coef_
        coef_df = pd.DataFrame({
            'feature': training_features,
            'coefficient': coefficients
        })
        # Add absolute coefficient for sorting/plotting
        coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)

        coef_path = model_dir / f"{model_prefix}_coefficients_{timestamp}.csv"
        coef_df.to_csv(coef_path, index=False)
        logger.info(f"ElasticNet coefficients saved: {coef_path}")

        # Log non-zero coefficients count (due to L1 part)
        non_zero_coefs = np.sum(coef_df['coefficient'] != 0)
        logger.info(f"Number of non-zero coefficients: {non_zero_coefs} / {len(training_features)}")

        logger.info("Top 20 Features by Absolute Coefficient:")
        logger.info("\n" + coef_df[['feature', 'coefficient']].head(20).to_string(index=False))

        # Coefficient Plot
        plt.figure(figsize=(10, 8))
        # Plot only top N coefficients by absolute value
        plot_data = coef_df.head(30)
        sns.barplot(x="coefficient", y="feature", data=plot_data, palette="vlag") # Use diverging palette
        plt.title(f"ElasticNet Top 30 Coefficients by Magnitude ({model_prefix.replace('_enet','')})")
        plt.tight_layout()
        plot_path = plot_dir / f"{model_prefix}_coef_imp_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"ElasticNet coefficient plot: {plot_path}")
        plt.close()

    except Exception as e:
        logger.error(f"Could not get/save ElasticNet coefficients: {e}")


    # --- Save Feature List ---
    # Save the FULL feature list that was actually used for training this model
    # ElasticNet doesn't generate a top-N list based on importance in the same way trees do
    features_path = model_dir / f"{model_prefix}_feature_columns_{timestamp}.pkl"
    try:
        with open(features_path, 'wb') as f: pickle.dump(training_features, f)
        logger.info(f"ElasticNet features list used for this run saved: {features_path}")
    except Exception as e: logger.error(f"Failed to save ElasticNet feature list: {e}")

    # --- Prediction vs Actual Plot ---
    try:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, test_preds, alpha=0.3, label='Test Set')
        plt.scatter(y_train_full, train_preds, alpha=0.05, label='Train Set', color='orange')
        min_val = min(min(y_test), min(y_train_full), 0) if not y_test.empty else 0
        max_val = max(max(y_test), max(y_train_full)) if not y_test.empty else max(y_train_full)
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label='Perfect Prediction')
        plt.xlabel("Actual Strikeouts"); plt.ylabel("Predicted Strikeouts")
        plt.title(f"Actual vs. Predicted Strikeouts (ElasticNet - {model_prefix.replace('_enet','')})")
        plt.legend(); plt.grid(True)
        plot_path = plot_dir / f"{model_prefix}_pred_actual_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"ElasticNet Pred/Actual plot: {plot_path}")
        plt.close()
    except Exception as e: logger.error(f"Failed to create ElasticNet pred/actual plot: {e}")

    # --- Save Model (using joblib) ---
    model_path = model_dir / f"{model_prefix}_strikeout_model_{timestamp}.joblib"
    try:
        joblib.dump(final_model, model_path)
        logger.info(f"ElasticNet model saved: {model_path}")
    except Exception as e: logger.error(f"Failed to save ElasticNet model: {e}")

    # --- SAVE THE SCALER ---
    scaler_path = model_dir / f"{model_prefix}_scaler_{timestamp}.joblib"
    try:
        joblib.dump(scaler, scaler_path)
        logger.info(f"StandardScaler saved: {scaler_path}")
    except Exception as e: logger.error(f"Failed to save StandardScaler: {e}")

    # --- Save best params (if found via Optuna) ---
    if should_run_optuna and best_params and study.best_value is not None:
         params_save_path_final = model_dir / f"{model_prefix}_best_params_{timestamp}.json"
         try:
             params_to_save = best_params.copy()
             params_to_save['best_cv_score_from_optuna'] = study.best_value
             with open(params_save_path_final, 'w') as f: json.dump(params_to_save, f, indent=4)
             logger.info(f"Saved final ElasticNet hyperparameters (from Optuna) to: {params_save_path_final}")
         except Exception as e: logger.error(f"Failed to save Optuna best parameters: {e}")
    elif best_params:
         logger.info(f"Final ElasticNet model trained using loaded parameters.")

    logger.info("ElasticNet model training finished successfully.")


# --- Main Execution Block ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    args = parse_args()
    train_model(args)
    logger.info("--- ElasticNet Training Script Completed ---")
