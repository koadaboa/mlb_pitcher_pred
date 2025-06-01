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
    from src.config import DBConfig, StrikeoutModelConfig, LogConfig, FileConfig
    from src.data.utils import setup_logger, DBConnection, find_latest_file
    # Import selection function AND BASE_EXCLUDE_COLS
    from src.features.selection import select_features, BASE_EXCLUDE_COLS
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    MODULE_IMPORTS_OK = False
    sys.exit(1)

LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger('train_lgb_model', LogConfig.LOG_DIR / 'train_lgb_model.log')
MODEL_TYPE = 'lgb'

# --- Evaluation Metric Helper --- (Keep as is)
def within_n_strikeouts(y_true, y_pred, n=1):
    """ Calculates percentage of predictions within n strikeouts """
    if y_true is None or y_pred is None or len(y_true) != len(y_pred): return np.nan
    y_true_arr = np.asarray(y_true); y_pred_arr = np.asarray(y_pred)
    return np.mean(np.abs(y_true_arr - np.round(y_pred_arr)) <= n)

# --- Optuna Objective Function --- (Keep as is)
def objective_lgbm_timeseries_poisson(trial, X_data, y_data):
    lgb_params = StrikeoutModelConfig.LGBM_BASE_PARAMS.copy()
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
    tscv = TimeSeriesSplit(n_splits=StrikeoutModelConfig.OPTUNA_CV_SPLITS)
    logger.debug(f"Trial {trial.number}: CV with {tscv.n_splits} splits...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_data)):
        # ... (rest of CV logic remains the same) ...
        X_train_fold, X_val_fold = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train_fold, y_val_fold = y_data.iloc[train_idx], y_data.iloc[val_idx]
        if len(X_val_fold) == 0: continue
        train_set = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_set = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_set)
        callbacks = [lgb.early_stopping(StrikeoutModelConfig.EARLY_STOPPING_ROUNDS // 2, verbose=False)]
        model = lgb.train(lgb_params, train_set,
                          num_boost_round=StrikeoutModelConfig.FINAL_ESTIMATORS,
                          valid_sets=[train_set, val_set],
                          callbacks=callbacks)
        preds = model.predict(X_val_fold, num_iteration=model.best_iteration)
        dev = mean_poisson_deviance(y_val_fold, preds)
        fold_devs.append(dev)
        logger.debug(f"  Trial {trial.number} Fold {fold+1}/{tscv.n_splits} - Val Poisson Deviance: {dev:.4f}")
        trial.report(dev, fold)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()
    if not fold_devs: return float('inf')
    return float(np.mean(fold_devs))

# --- Argument Parser (Keep as is, 'all' default removed as selection is now multi-stage) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM Strikeout Model")
    parser.add_argument("--mode", type=str, default="tune", choices=['tune', 'retrain', 'production'],
                        help="Training mode. Default: tune")
    # Feature selection now implies the multi-stage process if not 'none'
    parser.add_argument("--feature-selection", type=str, default="vif_shap",
                        choices=['none', 'vif', 'shap', 'vif_shap'], # Revert default or keep vif_shap? Let's keep vif_shap
                        help="Feature selection method: 'none', 'vif' (only VIF before Optuna), 'shap' (only SHAP after Optuna), 'vif_shap' (VIF before, SHAP after). Default: vif_shap")
    parser.add_argument("--optuna-trials", type=int, default=None, help="Override number of Optuna trials.")
    parser.add_argument("--top-n-features", type=int, default=None, help="Save list of top N features.")
    parser.add_argument("--prod-artifact", action="store_true", help="Save artifacts with 'prod_' prefix.")
    parser.add_argument("--verbose-fit", action="store_true", help="Enable verbose fitting output.")
    parser.add_argument("--vif-threshold", type=float, default=None,
                        help=f"VIF pruning threshold (default: {StrikeoutModelConfig.VIF_THRESHOLD})")
    parser.add_argument("--shap-threshold", type=float, default=None,
                        help=f"SHAP mean abs threshold (default: {StrikeoutModelConfig.SHAP_THRESHOLD})")
    return parser.parse_args()

# --- Main Training Function (REFACTORED for Multi-Stage Selection) ---
def train_model(args):
    logger.info(f"Starting LightGBM training (Mode: {args.mode}, Feature Selection: {args.feature_selection})...")
    db_path = Path(DBConfig.PATH)
    model_dir = Path(FileConfig.MODELS_DIR)
    plot_dir = Path(FileConfig.PLOTS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = f"prod_{MODEL_TYPE}" if args.prod_artifact else f"test_{MODEL_TYPE}"
    logger.info(f"Artifact prefix: {model_prefix}")
    target = StrikeoutModelConfig.TARGET_VARIABLE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Timestamp for artifacts

    # --- Load Data ---
    logger.info("Loading feature data...")
    # ... (Keep data loading logic using 'train_features', 'test_features') ...
    try:
        with DBConnection(db_path) as conn:
            train_table = "train_features"
            test_table = "test_features"
            train_df = pd.read_sql_query(f"SELECT * FROM {train_table}", conn)
            logger.info(f"Loaded {len(train_df)} train rows from '{train_table}'")
            test_df = pd.read_sql_query(f"SELECT * FROM {test_table}", conn)
            logger.info(f"Loaded {len(test_df)} test rows from '{test_table}'")
        logger.info(f"Available columns ({len(train_df.columns)}): {train_df.columns.tolist()}")
    except Exception as e: logger.error(f"Error loading feature data: {e}", exc_info=True); sys.exit(1)


    # --- Initial NaN/Inf Handling ---
    logger.info("Checking for infinite/NaN values in loaded data...")
    imputation_map = {}
    for df_name, df_obj in [('train_df', train_df), ('test_df', test_df)]:
        numeric_cols = df_obj.select_dtypes(include=np.number).columns.tolist()
        df_data = df_obj.replace([np.inf, -np.inf], np.nan)
        if numeric_cols:
            nan_check_series = df_data[numeric_cols].isnull().any()
            cols_with_nan = nan_check_series[nan_check_series].index.tolist()
            if cols_with_nan:
                logger.warning(f"NaN values found in {df_name}. Imputing with train median/0.")
                for col in cols_with_nan:
                    if col not in numeric_cols: continue
                    if df_name == 'train_df':
                        median_val = df_data[col].median()
                        fill_val = median_val if pd.notna(median_val) else 0
                        imputation_map[col] = fill_val
                    else: fill_val = imputation_map.get(col, 0)
                    df_data[col] = df_data[col].fillna(fill_val)
        # Assign potentially modified data back
        if df_name == 'train_df': train_df = df_data
        elif df_name == 'test_df': test_df = df_data

    # --- Define Initial Feature Set (All available numeric minus exclusions) ---
    logger.info("Defining initial feature set...")
    exclude_set = set(BASE_EXCLUDE_COLS) | {target}
    all_cols = train_df.columns.tolist()
    initial_features = [col for col in all_cols if col not in exclude_set and pd.api.types.is_numeric_dtype(train_df[col])]
    logger.info(f"Initial potential features ({len(initial_features)}): {sorted(initial_features)}")

    # --- Variables to store results ---
    best_params = None
    final_training_features = initial_features # Default if no selection runs
    prelim_model_trained = None # To hold model for SHAP

    # === Stage 1: Optuna Tuning (and optional Pre-VIF) ===
    if args.mode == 'tune':
        features_for_optuna = initial_features.copy()

        # Optional VIF Pruning BEFORE Optuna
        if args.feature_selection in ['vif', 'vif_shap']:
            logger.info("--- Running Pre-Optuna VIF Selection ---")
            features_for_optuna, _ = select_features(
                train_df.copy(), # Use cleaned train_df
                target_variable=target,
                exclude_cols=list(exclude_set | (set(all_cols) - set(initial_features))), # Exclude non-initial features
                prune_vif=True,
                vif_threshold=(args.vif_threshold
                               if args.vif_threshold is not None
                               else StrikeoutModelConfig.VIF_THRESHOLD),
                prune_shap=False # SHAP is done later
            )
            if not features_for_optuna: logger.error("No features left after VIF!"); sys.exit(1)
            logger.info(f"Features after VIF ({len(features_for_optuna)}): {sorted(features_for_optuna)}")
        else:
             logger.info("Skipping Pre-Optuna VIF selection based on args.")

        # Run Optuna
        logger.info(f"--- Running Optuna ({len(features_for_optuna)} features) ---")
        optuna_start_time = time.time()
        n_trials = args.optuna_trials if args.optuna_trials is not None else StrikeoutModelConfig.OPTUNA_TRIALS
        study = optuna.create_study(direction='minimize')
        try:
            study.optimize(lambda trial: objective_lgbm_timeseries_poisson(trial, train_df[features_for_optuna], train_df[target]),
                           n_trials=n_trials, timeout=StrikeoutModelConfig.OPTUNA_TIMEOUT)
            tuned_params = study.best_params
            # Combine with base params
            best_params = StrikeoutModelConfig.LGBM_BASE_PARAMS.copy()
            best_params.update(tuned_params)
            logger.info(f"Optuna finished in {(time.time() - optuna_start_time):.2f}s. Best CV Score: {study.best_value:.4f}")
            logger.info(f"Best hyperparameters found: {best_params}")
            # Save best params
            params_save_path = model_dir / f"{model_prefix}_best_params_{timestamp}.json"
            try:
                params_to_save = best_params.copy(); params_to_save['best_cv_score_from_optuna'] = study.best_value
                with open(params_save_path, 'w') as f: json.dump(params_to_save, f, indent=4)
                logger.info(f"Saved best hyperparameters to: {params_save_path}")
            except Exception as e: logger.error(f"Failed to save best params: {e}")
        except Exception as e:
             logger.error(f"Optuna optimization failed: {e}", exc_info=True); sys.exit(1)

        # === Stage 2: Preliminary Model Training (for SHAP) ===
        if args.feature_selection in ['shap', 'vif_shap']:
             logger.info(f"--- Training Preliminary Model ({len(features_for_optuna)} features for SHAP) ---")
             X_train_prelim = train_df[features_for_optuna]
             y_train_prelim = train_df[target]
             train_dataset_prelim = lgb.Dataset(X_train_prelim, label=y_train_prelim)
             try:
                 # Train a simple model without validation/early stopping for SHAP purposes
                 prelim_model_trained = lgb.train(best_params, train_dataset_prelim,
                                                  num_boost_round=StrikeoutModelConfig.FINAL_ESTIMATORS // 2) # Train for fewer rounds?
                 logger.info("Preliminary model training complete.")
             except Exception as e:
                 logger.error(f"Failed to train preliminary model for SHAP: {e}", exc_info=True)
                 prelim_model_trained = None # Ensure it's None if failed

        # === Stage 3: SHAP Pruning (if applicable) ===
        if args.feature_selection in ['shap', 'vif_shap'] and prelim_model_trained is not None:
            logger.info(f"--- Running Post-Optuna SHAP Selection ---")
            # Pass the preliminary model and the features it was trained on
            final_training_features, _ = select_features(
                train_df.copy(), # Use original cleaned DF
                target_variable=target,
                exclude_cols=list(exclude_set | (set(all_cols) - set(features_for_optuna))), # Exclude non-optuna features
                prune_vif=False, # VIF already done (or skipped)
                prune_shap=True,
                shap_model=prelim_model_trained, # Pass the trained model
                shap_threshold=(args.shap_threshold
                                if args.shap_threshold is not None
                                else StrikeoutModelConfig.SHAP_THRESHOLD),
                shap_sample_frac=StrikeoutModelConfig.SHAP_SAMPLE_FRAC
            )
            if not final_training_features: logger.error("No features left after SHAP!"); sys.exit(1)
            logger.info(f"Features after SHAP ({len(final_training_features)}): {sorted(final_training_features)}")
        elif prelim_model_trained is None and args.feature_selection in ['shap', 'vif_shap']:
             logger.warning("Preliminary model training failed. Skipping SHAP selection. Using features from previous step.")
             final_training_features = features_for_optuna # Fallback to features used for Optuna
        else:
            logger.info("Skipping Post-Optuna SHAP selection based on args or prelim model failure.")
            final_training_features = features_for_optuna # Use features from pre-SHAP step

        # --- Save the final selected feature list ---
        features_path = model_dir / f"{model_prefix}_feature_columns_{timestamp}.pkl"
        try:
            with open(features_path, 'wb') as f: pickle.dump(final_training_features, f)
            logger.info(f"Final features list used ({len(final_training_features)}) saved: {features_path}")
        except Exception as e: logger.error(f"Failed to save final feature list: {e}")

    # === End of TUNE mode specific logic ===

    # === Load Parameters and Features for Retrain/Production ===
    elif args.mode in ['retrain', 'production']:
        logger.info(f"Mode '{args.mode}': Loading latest parameters and feature list...")
        # Load best params
        param_pattern = f"{model_prefix}_best_params_*.json"
        latest_params_file = find_latest_file(model_dir, param_pattern)
        if latest_params_file:
            logger.info(f"Loading hyperparameters from: {latest_params_file}")
            try:
                with open(latest_params_file, 'r') as f: best_params = json.load(f)
                best_params.pop('best_iteration_from_final_train', None)
                best_params.pop('best_cv_score_from_optuna', None)
            except Exception as e: logger.error(f"Failed load params: {e}. Exiting."); sys.exit(1)
        else: logger.error(f"No param file found matching {param_pattern}. Cannot retrain/run production. Exiting."); sys.exit(1)

        # Load final features list
        feature_pattern = f"{model_prefix}_feature_columns_*.pkl"
        latest_feature_file = find_latest_file(model_dir, feature_pattern)
        if latest_feature_file:
            logger.info(f"Loading feature list from: {latest_feature_file}")
            try:
                with open(latest_feature_file, 'rb') as f: final_training_features = pickle.load(f)
                if not final_training_features: raise ValueError("Loaded feature list is empty.")
                logger.info(f"Loaded {len(final_training_features)} features.")
            except Exception as e: logger.error(f"Failed load feature list: {e}. Exiting."); sys.exit(1)
        else: logger.error(f"No feature list file found matching {feature_pattern}. Cannot retrain/run production. Exiting."); sys.exit(1)

    # === Safety check if best_params or final_training_features are still None ===
    if best_params is None: logger.error("Best parameters not loaded or tuned. Exiting."); sys.exit(1)
    if not final_training_features: logger.error("Final training feature list is empty. Exiting."); sys.exit(1)


    # --- Prepare FINAL Data Splits using final_training_features ---
    logger.info(f"Preparing final data splits using {len(final_training_features)} features...")
    X_train_full = train_df[final_training_features].copy()
    y_train_full = train_df[target].copy()
    X_test = test_df[[f for f in final_training_features if f in test_df.columns]].copy()
    y_test = test_df[target].copy()

    missing_test_cols = [f for f in final_training_features if f not in X_test.columns]
    if missing_test_cols:
        logger.warning(f"Final feature columns missing in test set: {missing_test_cols}. Adding as NaN & imputing.")
        for col in missing_test_cols:
            X_test[col] = np.nan
            fill_val = imputation_map.get(col, 0) # Use imputation map from start
            X_test[col] = X_test[col].fillna(fill_val)

    logger.info(f"Final Train shape: {X_train_full.shape}, Final Test shape: {X_test.shape}")
    del train_df, test_df; gc.collect()


    # --- Final Model Training ---
    logger.info("Preparing data for final model training...")
    # ... (Final training logic based on args.mode using final_training_features and best_params remains the same) ...
    if args.mode == 'production':
        logger.warning("Production mode: Training on ALL available data (Train + Test).")
        X_final_train = pd.concat([X_train_full, X_test], ignore_index=True)
        y_final_train = pd.concat([y_train_full, y_test], ignore_index=True)
        train_dataset = lgb.Dataset(X_final_train, label=y_final_train, feature_name=final_training_features)
        valid_sets = [train_dataset]; valid_names = ['train']
        early_stopping_rounds = None
    else:
        X_final_train = X_train_full; y_final_train = y_train_full
        train_dataset = lgb.Dataset(X_final_train, label=y_final_train, feature_name=final_training_features)
        valid_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)
        valid_sets = [train_dataset, valid_dataset]; valid_names = ['train', 'eval']
        early_stopping_rounds = StrikeoutModelConfig.EARLY_STOPPING_ROUNDS

    verbose_eval = StrikeoutModelConfig.VERBOSE_FIT_FREQUENCY if args.verbose_fit else 0
    callbacks = [lgb.log_evaluation(period=verbose_eval)]
    if early_stopping_rounds is not None and early_stopping_rounds > 0: callbacks.insert(0, lgb.early_stopping(early_stopping_rounds, verbose=args.verbose_fit))

    logger.info("Training final model...")
    try:
        final_model = lgb.train(
            best_params, train_dataset,
            num_boost_round=StrikeoutModelConfig.FINAL_ESTIMATORS,
            valid_sets=valid_sets, valid_names=valid_names,
            callbacks=callbacks )
        best_iter = final_model.best_iteration if early_stopping_rounds else StrikeoutModelConfig.FINAL_ESTIMATORS
        logger.info(f"Final model training complete. Best iteration: {best_iter}")
    except Exception as e: logger.error(f"Failed to train final model: {e}", exc_info=True); sys.exit(1)


    # --- Evaluation ---
    # ... (Evaluation logic remains the same, using final_training_features, final_model, best_iter) ...
    logger.info("Evaluating model...")
    train_preds = final_model.predict(X_final_train, num_iteration=best_iter)
    train_rmse = np.sqrt(mean_squared_error(y_final_train, train_preds))
    train_mae = mean_absolute_error(y_final_train, train_preds)
    train_pdev = mean_poisson_deviance(y_final_train, train_preds)
    logger.info("--- Train Metrics ---")
    logger.info(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
    logger.info(f"Train Poisson Deviance: {train_pdev:.4f}")

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
    else: logger.info("Test set not available for evaluation.")

    # --- Save Artifacts (including the FINAL feature list again) ---
    # ... (Saving logic remains the same, but ensure params_used includes info about selection stages) ...
    # Save Importance
    try:
        # Use final_training_features list which has correct length
        importance_df = pd.DataFrame({
            'feature': final_training_features,
            'importance': final_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        imp_path = model_dir / f"{model_prefix}_feature_importance_full_{timestamp}.csv"
        importance_df.to_csv(imp_path, index=False)
        logger.info(f"Full importance list saved: {imp_path}")
        logger.info("--- Top 10 Feature Importances (Gain) ---")
        logger.info("\n" + importance_df.head(10).to_string(index=False))
        logger.info("------------------------------------------")
    except Exception as e: logger.error(f"Could not get/save/log feature importances: {e}")

    # Save plots (using final data/preds)
    try: plt.figure(figsize=(10, max(8, len(final_training_features)//5))); plot_n = min(30, len(importance_df)); sns.barplot(x="importance", y="feature", data=importance_df.head(plot_n)); plt.title(f"LGBM Feature Importance ({model_prefix.replace('_lgb','')} - Top {plot_n})"); plt.tight_layout(); plot_path = plot_dir / f"{model_prefix}_feat_imp_{timestamp}.png"; plt.savefig(plot_path); logger.info(f"Importance plot: {plot_path}"); plt.close()
    except Exception as e: logger.error(f"Failed to create importance plot: {e}")
    try: plt.figure(figsize=(8, 8)); plt.scatter(y_test, test_preds, alpha=0.3, label='Test Set'); plt.scatter(y_final_train, train_preds, alpha=0.05, label='Train Set (Final)', color='orange'); min_val=min(y_test.min(), y_final_train.min()); max_val=max(y_test.max(), y_final_train.max()); plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label='Perfect Prediction'); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(f"Actual vs. Predicted (LGBM - {model_prefix.replace('_lgb','')})"); plt.legend(); plt.grid(True); plot_path = plot_dir / f"{model_prefix}_pred_actual_{timestamp}.png"; plt.savefig(plot_path); logger.info(f"Pred/Actual plot: {plot_path}"); plt.close()
    except Exception as e: logger.error(f"Failed to create pred/actual plot: {e}")

    # Save Top N Features List (if requested)
    # ... (save top_n from importance_df) ...

    # Save FINAL feature list used
    features_path = model_dir / f"{model_prefix}_feature_columns_{timestamp}.pkl" # Overwrite previous save with FINAL list
    try:
        with open(features_path, 'wb') as f: pickle.dump(final_training_features, f)
        logger.info(f"FINAL Features list used ({len(final_training_features)}) saved: {features_path}")
    except Exception as e: logger.error(f"Failed to save FINAL feature list: {e}")

    # Save Model
    model_path = model_dir / f"{model_prefix}_strikeout_model_{timestamp}.txt"
    final_model.save_model(str(model_path))
    logger.info(f"Model saved: {model_path}")

    # Save Params Used
    params_used_path = model_dir / f"{model_prefix}_params_used_{timestamp}.json"
    try:
        params_to_save = best_params.copy()
        params_to_save['best_iteration_used'] = best_iter
        if args.mode == 'tune' and 'study' in locals(): params_to_save['best_cv_score_from_optuna'] = study.best_value
        params_to_save['feature_selection_method'] = args.feature_selection # Record method used
        with open(params_used_path, 'w') as f: json.dump(params_to_save, f, indent=4)
        logger.info(f"Parameters used for training saved: {params_used_path}")
    except Exception as e: logger.error(f"Failed to save parameters used: {e}")

    logger.info("Model training finished successfully.")


# --- Main Execution Block ---
if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    args = parse_args()
    train_model(args)
    logger.info("--- LightGBM Training Script Completed ---")