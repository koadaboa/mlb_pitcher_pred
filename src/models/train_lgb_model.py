# src/models/train_lgb_model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import joblib # Or use pickle
import pickle
import json
import argparse
import logging
import sys
import time
import re # For parsing filenames
import os # For path operations
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit # Import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import gc


# Assuming script is run via python -m src.models.train_lgb_model
from src.config import DBConfig, StrikeoutModelConfig, LogConfig, FileConfig
from src.data.utils import setup_logger, DBConnection

# Setup logger
LogConfig.LOG_DIR.mkdir(parents=True, exist_ok=True) # Ensure log dir exists
logger = setup_logger('train_lgb_model', LogConfig.LOG_DIR / 'train_lgb_model.log')

# --- Helper Function to Find Latest File ---
def find_latest_file(directory, pattern):
    """
    Finds the most recent file in a directory matching a glob pattern,
    based on timestamp in the filename or modification time.
    """
    model_dir = Path(directory)
    if not model_dir.is_dir(): logger.error(f"Directory not found: {model_dir}"); return None
    files = list(model_dir.glob(pattern))
    if not files: logger.warning(f"No files found matching pattern '{pattern}' in {model_dir}"); return None
    latest_file, latest_timestamp = None, 0
    ts_pattern = re.compile(r"_(\d{8}_\d{6})\.") # Matches _YYYYMMDD_HHMMSS.
    parsed_successfully = False
    for f in files:
        match = ts_pattern.search(f.name)
        if match:
            try:
                ts = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").timestamp()
                if ts > latest_timestamp: latest_timestamp, latest_file = ts, f
                parsed_successfully = True
            except ValueError: logger.warning(f"Could not parse timestamp from: {f.name}"); pass
    if not parsed_successfully and files:
        logger.warning(f"Could not determine latest file by timestamp for '{pattern}'. Falling back to mtime.")
        try: latest_file = max(files, key=lambda x: x.stat().st_mtime); parsed_successfully = True
        except Exception as e: logger.error(f"Error finding latest file by mtime for {pattern}: {e}"); return None
    if latest_file: logger.info(f"Found latest file for pattern '{pattern}': {latest_file.name}")
    elif not parsed_successfully: logger.error(f"Could not find latest file for pattern '{pattern}' using any method.")
    return latest_file

# --- Argument Parser ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train LightGBM Strikeout Model")
    parser.add_argument("--optuna-trials", type=int, default=StrikeoutModelConfig.OPTUNA_TRIALS, help="Number of Optuna trials.")
    parser.add_argument("--optuna-timeout", type=int, default=StrikeoutModelConfig.OPTUNA_TIMEOUT, help="Optuna timeout in seconds.")
    # ADD the --use-best-params flag
    parser.add_argument("--use-best-params", action="store_true",
                        help="Automatically load the latest saved best parameters for LightGBM (skips Optuna).")
    # REMOVE --params-file if it existed
    parser.add_argument("--production", action="store_true", help="Train final model on all data (no Optuna).")
    parser.add_argument("--final-estimators", type=int, default=StrikeoutModelConfig.FINAL_ESTIMATORS, help="Max estimators for final model.")
    parser.add_argument("--early-stopping", type=int, default=StrikeoutModelConfig.EARLY_STOPPING_ROUNDS, help="Early stopping rounds for final model.")
    parser.add_argument("--verbose-fit", action="store_true", help="Enable verbose fitting output.")
    return parser.parse_args()

# --- Evaluation Metric Helper ---
def within_n_strikeouts(y_true, y_pred, n=1):
    """ Calculates percentage of predictions within n strikeouts """
    if y_true is None or y_pred is None or len(y_true) != len(y_pred): return np.nan
    # Ensure numpy arrays for vectorized operation
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return np.mean(np.abs(y_true_arr - np.round(y_pred_arr)) <= n)

# --- Optuna Objective Function (using TimeSeriesSplit) ---
# Define this function *before* it's called in train_model
def objective_lgbm_timeseries_rmse(trial, X_data, y_data):
    """ Optuna objective function using TimeSeriesSplit CV for LightGBM """
    # Define search space
    lgbm_params = {
        'objective': 'regression_l1', # MAE objective
        'metric': 'rmse', # Still evaluate on RMSE
        'boosting_type': 'gbdt', 'feature_pre_filter': False, 'n_jobs': -1,
        'verbose': -1, 'seed': StrikeoutModelConfig.RANDOM_STATE,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    fold_rmses = []
    n_cv_splits = 4 # Adjust as needed
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)

    logger.debug(f"Trial {trial.number}: Starting TimeSeriesSplit CV with {n_cv_splits} splits...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_data)):
        X_train_fold, X_val_fold = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train_fold, y_val_fold = y_data.iloc[train_idx], y_data.iloc[val_idx]
        if len(X_val_fold) == 0: continue

        lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
        lgb_val = lgb.Dataset(X_val_fold, label=y_val_fold, reference=lgb_train)

        model = lgb.train(lgbm_params, lgb_train, num_boost_round=2000, # Max rounds for tuning
                          valid_sets=[lgb_val], valid_names=['validation'],
                          callbacks=[lgb.early_stopping(100, verbose=False)]) # Early stopping per fold

        preds = model.predict(X_val_fold, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        fold_rmses.append(rmse)
        logger.debug(f"  Trial {trial.number} Fold {fold+1}/{n_cv_splits} - Val RMSE: {rmse:.4f}")

    if not fold_rmses: return float('inf')
    average_rmse = np.mean(fold_rmses)
    logger.debug(f"Trial {trial.number} completed. Average CV RMSE: {average_rmse:.4f}")
    return average_rmse

# --- Feature Selection ---
def select_features(df, target_variable = 'strikeout'):
    """Selects numeric features, excluding specified columns."""
    if df is None or df.empty:
        return [], df

    exclude_cols = [
        # Identifiers / Non-Features
        'index', '', 'pitcher_id', 'player_name', 'game_pk', 'home_team', 'away_team',
        'opponent_team_name', 'game_date', 'season', 'game_month', 'year',
        'p_throws', 'stand', 'team', 'Team', 'opponent_team', # Exclude original categoricals now that encoded versions exist
        'opp_base_team', 'opp_adv_team', 'opp_adv_opponent',

        # Target Variable
        'strikeouts',

        # --- DIRECT LEAKAGE COLUMNS (Derived from target game outcome/process) ---
        'batters_faced',            # Target game outcome
        'total_pitches',            # Target game outcome
        'innings_pitched',          # Target game outcome
        'avg_velocity',             # Target game outcome (use lagged/rolling instead)
        'max_velocity',             # Target game outcome (use lagged/rolling instead)
        'avg_spin_rate',            # Target game outcome (use lagged/rolling instead)
        'avg_horizontal_break',     # Target game outcome (use lagged/rolling instead)
        'avg_vertical_break',       # Target game outcome (use lagged/rolling instead)
        'k_per_9',                  # Target game outcome (derived from strikeouts)
        'k_percent',                # Target game outcome (derived from strikeouts)
        'swinging_strike_percent',  # Target game outcome (derived from total_swinging_strikes/total_pitches)
        'called_strike_percent',    # Target game outcome (derived from total_called_strikes/total_pitches)
        'zone_percent',             # Target game outcome (derived from total_in_zone/total_pitches)
        'fastball_percent',         # Target game outcome (derived from total_fastballs/total_pitches)
        'breaking_percent',         # Target game outcome (derived from total_breaking/total_pitches)
        'offspeed_percent',         # Target game outcome (derived from total_offspeed/total_pitches)
        'total_swinging_strikes',   # Target game count
        'total_called_strikes',     # Target game count
        'total_fastballs',          # Target game count
        'total_breaking',           # Target game count
        'total_offspeed',           # Target game count
        'total_in_zone',            # Target game count
        'pa_vs_rhb',                # Target game platoon outcome
        'k_vs_rhb',                 # Target game platoon outcome
        'k_percent_vs_rhb',         # Target game platoon outcome
        'pa_vs_lhb',                # Target game platoon outcome
        'k_vs_lhb',                 # Target game platoon outcome
        'k_percent_vs_lhb',         # Target game platoon outcome
        'platoon_split_k_pct',      # Target game platoon outcome
        # -----------------------------------------------------------------------

        # Other potential post-game info or less relevant features
        'inning', 'score_differential', 'is_close_game', 'is_playoff',

        # Low importance / redundant features (review based on new importance)
        'is_home', # Encoded version exists
        'rest_days_6_more', 'rest_days_4_less', 'rest_days_5', # days_since_last_game is likely better
        'is_month_3', 'is_month_4', 'is_month_5', 'is_month_6', 'is_month_7', 'is_month_8', 'is_month_9', 'is_month_10',
        'throws_right', # Encoded version exists

        # Older opponent features (replace with opp_roll_*) - KEEP EXCLUDED
        'opp_adv_rolling_10g_team_batting_k_pct_vs_hand_R',
        'opp_adv_rolling_10g_team_batting_k_pct_vs_hand_R_std',
        'opp_adv_rolling_10g_team_batting_k_pct_vs_hand_L_std',
        'opp_adv_rolling_10g_team_batting_woba_vs_hand_R',
        'opp_adv_rolling_10g_team_batting_woba_vs_hand_R_std',
        'opp_adv_rolling_10g_team_batting_woba_vs_hand_L',
        'opp_adv_rolling_10g_team_batting_woba_vs_hand_L_std',
        'opp_adv_rolling_10g_team_pitching_k_pct_vs_batter_R',
        'opp_adv_rolling_10g_team_pitching_k_pct_vs_batter_R_std',
        'opp_adv_rolling_10g_team_pitching_k_pct_vs_batter_L',
        'opp_adv_rolling_10g_team_pitching_k_pct_vs_batter_L_std',
        'opp_base_team_wcb_c', 'opp_base_team_o_swing_percent',
        'opp_base_team_wsf_c', 'opp_base_team_wfb_c', 'opp_base_team_zone_percent', 'opp_base_team_wsl_c',
        'opp_base_team_wch_c', 'opp_base_team_k_percent', 'opp_base_team_wct_c', 'opp_base_team_z_contact_percent',
        'opp_base_team_bb_k_ratio', 'opp_base_team_contact_percent', 'opp_base_team_swstr_percent',

        # Other previously excluded low importance features (review later)
        'rolling_5g_fastball_pct', 'rolling_3g_fastball_pct', 'rolling_5g_k9', 'ewma_10g_avg_velocity_lag1',
        'rolling_10g_velocity', 'rolling_3g_breaking_pct', 'ewma_5g_k_per_9_lag1', 'ewma_5g_k_percent_lag1',
        'rolling_3g_breaking_percent_std_lag1', 'rolling_10g_k9', 'lag_1_breaking_percent',
        'lag_2_breaking_percent',
        'ewma_10g_offspeed_percent_lag1', 'offspeed_percent_change_lag1', 'rolling_5g_breaking_pct',
        'rolling_5g_offspeed_pct', 'rolling_3g_offspeed_pct', 'lag_1_offspeed_percent',
        'lag_2_offspeed_percent',

        # Imputation flags (usually not predictive)
        'avg_velocity_imputed_median', 'max_velocity_imputed_median', 'avg_spin_rate_imputed_median',
        'avg_horizontal_break_imputed_median', 'avg_vertical_break_imputed_median',
        'avg_velocity_imputed_knn', 'avg_spin_rate_imputed_knn',
        'avg_horizontal_break_imputed_knn', 'avg_vertical_break_imputed_knn',
        'last_game_strikeouts', 'strikeout_change', 'strikeouts_lag1', 'strikeouts_change',
        'batters_faced_change', 'k_per_9_pct_change', 'k_per_9_lag1', 'batters_faced_lag1', 'k_per_9_change',
        'innings_pitched_change', 'ewma_3g_strikeouts',
    ]

    # Select numeric columns only, excluding the ones in the list
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    logger.info(f"Selected {len(feature_cols)} numeric features.")
    logger.debug(f"Excluded columns: {exclude_cols}")
    logger.debug(f"First 5 selected features: {feature_cols[:5]}")

    # Ensure TARGET_VARIABLE exists before trying to select it
    return df[feature_cols].copy(), target_variable # Return a copy

# --- Main Training Function ---
def train_model(args):
    """Loads data, trains LightGBM model (with optional Optuna), evaluates, and saves."""
    logger.info(f"Starting training (Production Mode: {args.production})...")
    db_path = Path(DBConfig.PATH)
    model_dir = Path(FileConfig.MODELS_DIR) # Use config for model dir
    plot_dir = Path(FileConfig.PLOTS_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Data ---
    logger.info("Loading final feature data...")
    try:
        with DBConnection(db_path) as conn:
            # Load from the advanced tables
            train_df = pd.read_sql_query(f"SELECT * FROM train_features_advanced", conn)
            logger.info(f"Loaded {len(train_df)} train rows from 'train_features_advanced'")
            test_df = pd.read_sql_query(f"SELECT * FROM test_features_advanced", conn)
            logger.info(f"Loaded {len(test_df)} test rows from 'test_features_advanced'")
        all_data = pd.concat([train_df, test_df], ignore_index=True)
        logger.info(f"Total rows loaded: {len(all_data)}")
        logger.info(f"Available columns ({len(all_data.columns)}): {all_data.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error loading feature data: {e}", exc_info=True); sys.exit(1)

    # --- Feature Selection ---
    logger.info("Imputing missing values if any...")
    numeric_cols = all_data.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if all_data[col].isnull().any():
            median_val = all_data[col].median()
            logger.warning(f"Filling {all_data[col].isnull().sum()} NaNs in '{col}' with median {median_val}")
            all_data[col].fillna(median_val, inplace=True)

    # Select features using the function
    features_df, target_col = select_features(all_data, target_variable=StrikeoutModelConfig.TARGET_VARIABLE)
    training_features = list(features_df.columns) # Get list of feature names
    logger.info(f"Using {len(training_features)} features after exclusion and dtype filtering.")
    logger.info(f"Final features used: {training_features[:10]}...") # Log first few

    # --- Prepare Data Splits ---
    train_indices = all_data[all_data['season'].isin(StrikeoutModelConfig.DEFAULT_TRAIN_YEARS)].index
    test_indices = all_data[all_data['season'].isin(StrikeoutModelConfig.DEFAULT_TEST_YEARS)].index

    # Select only the final features for X
    X_train_full = features_df.loc[train_indices]
    y_train_full = all_data.loc[train_indices, target_col]
    X_test = features_df.loc[test_indices]
    y_test = all_data.loc[test_indices, target_col]

    logger.info(f"Train shape: {X_train_full.shape}, Test shape: {X_test.shape}")
    del all_data, features_df; gc.collect() # Memory management

    # ========================================================================
    # START: Corrected Hyperparameter Loading / Optuna Trigger Logic
    # ========================================================================
    best_params = None # Initialize best_params
    should_run_optuna = False # Flag to control Optuna execution

    # --- Determine if Optuna should run ---
    if args.production:
        logger.info("--- Running in PRODUCTION MODE ---")
        logger.info("Attempting to load latest best parameters for production...")
        should_run_optuna = False # Never run Optuna in production mode
        # Define pattern for LightGBM best params (adjust if your naming differs)
        param_pattern = "lgb_best_params_*.json" # Or "test_best_params_*.json"? Check naming
        latest_params_file = find_latest_file(model_dir, param_pattern)
        if latest_params_file:
            logger.info(f"Loading hyperparameters from: {latest_params_file}")
            try:
                with open(latest_params_file, 'r') as f: best_params = json.load(f)
                logger.info(f"Successfully loaded parameters for production: {best_params}")
            except Exception as e:
                logger.error(f"Failed to load prod parameters from {latest_params_file}: {e}. Using defaults.")
                best_params = {} # Use empty dict to trigger defaults
        else:
            logger.warning(f"No parameter file found matching '{param_pattern}' for production. Using defaults.")
            best_params = {} # Use empty dict to trigger defaults

    elif args.use_best_params:
        logger.info("Attempting to load latest best parameters (--use-best-params specified)...")
        should_run_optuna = False # Assume we won't run Optuna if flag is set
        param_pattern = "lgb_best_params_*.json" # Adjust pattern if needed
        latest_params_file = find_latest_file(model_dir, param_pattern)
        if latest_params_file:
            logger.info(f"Loading hyperparameters from: {latest_params_file}")
            try:
                with open(latest_params_file, 'r') as f: best_params = json.load(f)
                logger.info(f"Successfully loaded parameters: {best_params}")
            except Exception as e:
                logger.error(f"Failed to load parameters from {latest_params_file}: {e}. Will run Optuna.")
                best_params = None # Set to None to trigger Optuna
                should_run_optuna = True # Need to run Optuna now
        else:
            logger.warning(f"No best parameter file found matching '{param_pattern}'. Will run Optuna.")
            best_params = None # Set to None to trigger Optuna
            should_run_optuna = True # Need to run Optuna now
    else:
         # Not production and --use-best-params not set, run Optuna
         logger.info("No parameters loaded or specified (--use-best-params not set). Running Optuna.")
         should_run_optuna = True
         best_params = None # Ensure best_params is None before Optuna

    # --- Run Optuna Tuning ONLY if should_run_optuna is True ---
    if should_run_optuna:
        logger.info("Running Optuna hyperparameter search using TimeSeriesSplit CV for RMSE...")
        optuna_start_time = time.time()

        study = optuna.create_study(direction='minimize')
        try:
            # Pass the FULL training data to the objective function
            # Ensure objective_lgbm_timeseries_rmse is defined above
            study.optimize(lambda trial: objective_lgbm_timeseries_rmse(trial, X_train_full, y_train_full),
                           n_trials=args.optuna_trials, timeout=args.optuna_timeout)

            best_params = study.best_params # Get params from study
            logger.info(f"Optuna finished in {(time.time() - optuna_start_time):.2f}s. Best Average CV RMSE: {study.best_value:.4f}")
            logger.info(f"Best hyperparameters found: {best_params}")

            # Save the newly found best parameters
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_prefix = "lgb" # Optuna runs in non-production mode
            params_save_path = model_dir / f"lgb_best_params_{timestamp}.json" # Use specific prefix
            try:
                with open(params_save_path, 'w') as f: json.dump(best_params, f, indent=4)
                logger.info(f"Saved best hyperparameters to: {params_save_path}")
            except Exception as e: logger.error(f"Failed to save best parameters: {e}")

        except Exception as e:
             logger.error(f"Optuna optimization failed: {e}", exc_info=True)
             logger.warning("Proceeding with default LightGBM parameters after Optuna failure.")
             best_params = {} # Fallback to defaults if Optuna crashes

    # If params were loaded using --use-best-params, log them here
    elif best_params is not None: # Check if params were loaded from file
        logger.info(f"Using loaded hyperparameters: {best_params}")

    # --- Final Model Training ---
    # Use default params if tuning wasn't run or failed and no file loaded
    if best_params is None: best_params = {} # Ensure dict

    # Set defaults if params are empty (e.g., Optuna failed AND no file loaded/specified)
    if not best_params:
         logger.warning("No valid hyperparameters found/loaded. Using default LightGBM parameters.")
         # Define your default parameters here
         best_params = {'objective':'regression_l1', 'metric':'rmse', 'boosting_type':'gbdt', 'n_jobs': -1, 'verbose': -1, 'seed': StrikeoutModelConfig.RANDOM_STATE, 'learning_rate': 0.02, 'num_leaves': 50, 'max_depth': 7, 'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'bagging_freq': 3, 'reg_alpha': 0.1, 'reg_lambda': 0.1}

    # Ensure core params exist if loaded from file or from Optuna
    best_params.setdefault('objective','regression_l1'); best_params.setdefault('metric','rmse'); best_params.setdefault('seed', StrikeoutModelConfig.RANDOM_STATE); best_params.setdefault('n_jobs',-1); best_params.setdefault('verbose',-1);
    # ========================================================================
    # END: Corrected Hyperparameter Loading / Optuna Trigger Logic
    # ========================================================================

    logger.info("Training final model on full training set (using early stopping)...")
    lgb_train = lgb.Dataset(X_train_full, label=y_train_full)
    lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

    evals_result = {}
    final_model = lgb.train(best_params,
                            lgb_train,
                            num_boost_round=args.final_estimators,
                            valid_sets=[lgb_train, lgb_eval],
                            valid_names=['train', 'eval'],
                            callbacks=[lgb.early_stopping(args.early_stopping, verbose=args.verbose_fit),
                                       lgb.log_evaluation(period=100 if args.verbose_fit else 0),
                                       lgb.record_evaluation(evals_result)]) # Record evals

    logger.info(f"Final model training complete. Best iteration: {final_model.best_iteration}")

    # --- Evaluation ---
    logger.info(f"Evaluating on Test Set ({len(X_test)} rows)...")
    test_preds = final_model.predict(X_test, num_iteration=final_model.best_iteration)
    # Also get train predictions for comparison
    train_preds = final_model.predict(X_train_full, num_iteration=final_model.best_iteration)

    train_rmse = np.sqrt(mean_squared_error(y_train_full, train_preds))
    train_mae = mean_absolute_error(y_train_full, train_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    test_w1 = within_n_strikeouts(y_test, test_preds, n=1)
    test_w2 = within_n_strikeouts(y_test, test_preds, n=2)

    logger.info("--- Train Metrics ---")
    logger.info(f"Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f}")
    logger.info("--- Test Metrics ---")
    logger.info(f"Test RMSE : {test_rmse:.4f}, Test MAE : {test_mae:.4f}")
    logger.info(f"Test W/1 K: {test_w1:.4f}, Test W/2 K: {test_w2:.4f}")


    # --- Save Artifacts ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_prefix = "prod" if args.production else "test" # Use prefix for saved files

    # Feature Importance
    importance_df = pd.DataFrame({
        'feature': final_model.feature_name(),
        'importance': final_model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    imp_path = model_dir / f"{model_prefix}_feature_importance_full_{timestamp}.csv"
    importance_df.to_csv(imp_path, index=False)
    logger.info(f"Full importance list saved: {imp_path}")
    logger.info("Top 20 Feature Importances:")
    logger.info("\n" + importance_df.head(20).to_string(index=False))

    # Importance Plot
    try:
        plt.figure(figsize=(10, 8))
        sns.barplot(x="importance", y="feature", data=importance_df.head(30)) # Plot top 30
        plt.title(f"LightGBM Feature Importance ({model_prefix})")
        plt.tight_layout()
        plot_path = plot_dir / f"{model_prefix}_feat_imp_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"Importance plot: {plot_path}")
        plt.close()
    except Exception as e: logger.error(f"Failed to create importance plot: {e}")

    # Prediction vs Actual Plot
    try:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, test_preds, alpha=0.3, label='Test Set')
        # Add train predictions for comparison
        plt.scatter(y_train_full, train_preds, alpha=0.05, label='Train Set', color='orange')
        min_val = min(min(y_test), min(y_train_full), 0)
        max_val = max(max(y_test), max(y_train_full))
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label='Perfect Prediction')
        plt.xlabel("Actual Strikeouts")
        plt.ylabel("Predicted Strikeouts")
        plt.title(f"Actual vs. Predicted Strikeouts ({model_prefix})")
        plt.legend()
        plt.grid(True)
        plot_path = plot_dir / f"{model_prefix}_pred_actual_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"Pred/Actual plot: {plot_path}")
        plt.close()
    except Exception as e: logger.error(f"Failed to create pred/actual plot: {e}")

    # Save Model
    model_path = model_dir / f"{model_prefix}_strikeout_model_{timestamp}.txt"
    final_model.save_model(str(model_path))
    logger.info(f"Model saved: {model_path}")

    # Save Feature List (using pickle)
    features_path = model_dir / f"{model_prefix}_feature_columns_{timestamp}.pkl"
    try:
        with open(features_path, 'wb') as f:
            pickle.dump(training_features, f)
        logger.info(f"Features saved: {features_path}")
    except Exception as e: logger.error(f"Failed to save feature list: {e}")

    # Save best params if they came from Optuna (even if loaded, save with new timestamp?)
    # Let's only save if Optuna ran and found params
    if should_run_optuna and best_params and study.best_value is not None:
         params_save_path_final = model_dir / f"{model_prefix}_best_params_{timestamp}.json"
         try:
             # Add best iteration and score to saved params
             params_to_save = best_params.copy()
             params_to_save['best_iteration_from_optuna_cv'] = final_model.best_iteration # Iteration from final train
             params_to_save['best_cv_score_from_optuna'] = study.best_value
             with open(params_save_path_final, 'w') as f: json.dump(params_to_save, f, indent=4)
             logger.info(f"Saved final best hyperparameters (from Optuna run) to: {params_save_path_final}")
         except Exception as e: logger.error(f"Failed to save Optuna best parameters: {e}")
    elif best_params: # Log loaded params were used
         logger.info(f"Final model trained using parameters loaded via --use-best-params or for production.")


    logger.info("Model training finished successfully.")


if __name__ == "__main__":
    args = parse_args()
    # --- Ensure Objective Function is Defined ---
    # (Copy the objective_lgbm_timeseries_rmse function definition here if not imported)
    train_model(args) # Call the main training function
    logger.info("--- LightGBM Training Script Completed ---")



