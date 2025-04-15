# src/models/train_complete_model.py
# Rewritten to optimize for W1 Accuracy using Time-Series CV

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split # Still used for final early stopping
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import os
import sys
import pickle
from pathlib import Path
import argparse
import time
import json

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Attempt imports
try:
    from src.data.utils import setup_logger, DBConnection
    from src.config import StrikeoutModelConfig, DBConfig
    MODULE_IMPORTS_OK = True
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    MODULE_IMPORTS_OK = False
    # Fallback definitions... (keep as is)
    def setup_logger(name, level=logging.INFO, log_file=None): logging.basicConfig(level=level); return logging.getLogger(name)
    class DBConnection:
        def __init__(self, p): self.p=p; print(f"WARN: Using dummy DBConnection for {p}")
        def __enter__(self): return None
        def __exit__(self,t,v,tb): pass
    class StrikeoutModelConfig: RANDOM_STATE=42; DEFAULT_TRAIN_YEARS=(); DEFAULT_TEST_YEARS=(); WINDOW_SIZES=()
    class DBConfig: PATH="data/pitcher_stats.db"


# Setup logger
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True, parents=True)
logger = setup_logger('train_complete_model', log_file= log_dir / 'train_model.log', level=logging.INFO) if MODULE_IMPORTS_OK else logging.getLogger('train_fallback')

# --- Helper Function (Identical) ---
def within_n_strikeouts(y_true, y_pred, n=1):
    """ Calculates percentage of predictions within n strikeouts """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    within_n = np.abs(y_true - np.round(y_pred)) <= n
    return np.mean(within_n)

# --- Main Training Function ---
def train_model(production_mode=False, hyperparams=None):
    db_path = project_root / DBConfig.PATH
    logger.info("Loading combined predictive feature data...")
    all_historical_data = []
    try:
        # Data Loading (Identical)
        with DBConnection(db_path) as conn:
            if conn is None: raise ConnectionError("DB Connection failed.")
            try:
                train_df_load=pd.read_sql_query("SELECT * FROM train_combined_features", conn); logger.info(f"Loaded {len(train_df_load)} train rows")
                if not train_df_load.empty: all_historical_data.append(train_df_load)
            except Exception as e: logger.warning(f"No train_combined_features: {e}")
            try:
                test_df_load=pd.read_sql_query("SELECT * FROM test_combined_features", conn); logger.info(f"Loaded {len(test_df_load)} test rows")
                if not test_df_load.empty: all_historical_data.append(test_df_load)
            except Exception as e: logger.warning(f"No test_combined_features: {e}")
        if not all_historical_data: logger.error("No historical data loaded."); return None, None
        df = pd.concat(all_historical_data, ignore_index=True); logger.info(f"Total rows: {len(df)}")
    except Exception as e: logger.error(f"Failed load features: {e}", exc_info=True); return None, None

    # --- Data Prep (Identical) ---
    df['game_date'] = pd.to_datetime(df['game_date']).dt.date; df = df.sort_values(['pitcher_id', 'game_date'])
    numeric_cols = df.select_dtypes(include=np.number).columns; imputation_values = {}; logger.info("Imputing missing values...")
    for col in numeric_cols:
        if df[col].isnull().any(): median_val = df[col].median(); df[col]=df[col].fillna(median_val); imputation_values[col]=median_val

    # --- Feature Selection (Identical - ensure exclude_cols is up-to-date) ---
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
        'rolling_3g_breaking_percent_std_lag1', 'rolling_10g_k9', 'lag_1_breaking_percent', 
        'opp_adv_rolling_10g_team_batting_k_pct_vs_hand_L', 'lag_2_breaking_percent', 
        'ewma_10g_offspeed_percent_lag1', 'offspeed_percent_change_lag1', 'rolling_5g_breaking_pct', 
        'rolling_5g_offspeed_pct', 'rolling_3g_offspeed_pct', 'lag_1_offspeed_percent', 
        'opp_base_team_wcb_c', 'opp_base_team_o_swing_percent', 'lag_2_offspeed_percent', 
        'opp_base_team_wsf_c', 'opp_base_team_wfb_c', 'opp_base_team_zone_percent', 'opp_base_team_wsl_c', 
        'opp_base_team_wch_c', 'opp_base_team_k_percent', 'opp_base_team_wct_c', 'opp_base_team_z_contact_percent', 
        'opp_base_team_bb_k_ratio', 'opp_base_team_contact_percent', 'opp_base_team_swstr_percent', 'throws_right'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.int64, np.float64, 'int', 'float', 'int32', 'float32', 'Int64']]
    logger.info(f"Using {len(feature_cols)} features.")
    if len(feature_cols) < 5: logger.warning(f"Low feature count: {feature_cols}.")


    # --- Mode Logic ---
    final_model = None; best_params = hyperparams

    if production_mode:
        # --- Production Mode (Identical) ---
        logger.info("--- Running in PRODUCTION MODE ---")
        # (Keep the production mode logic exactly as in your provided script)
        X_train_all = df[feature_cols]; y_train_all = df['strikeouts']
        if X_train_all.empty: logger.error("No training data."); return None, None
        logger.info(f"Training production model on {len(X_train_all)} samples.")
        if best_params is None:
            logger.warning("No hyperparameters provided for production. Using defaults.")
            # Define defaults, potentially based on previous best W1 results or reasonable values
            best_params = {'objective':'regression', 'metric':'rmse', 'boosting_type':'gbdt', 'learning_rate': 0.03, 'num_leaves': 40, 'max_depth': 6, 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'verbose': -1, 'n_jobs': -1, 'seed': StrikeoutModelConfig.RANDOM_STATE}
        else: logger.info(f"Using provided hyperparameters for production: {best_params}")
        best_params.setdefault('objective','regression'); best_params.setdefault('metric','rmse'); # Use regression/rmse as base
        best_params.setdefault('verbose',-1); best_params.setdefault('n_jobs',-1);
        num_boost_round = best_params.pop('num_boost_round', 1000) # Allow override
        logger.info(f"Training for max {num_boost_round} rounds (no early stopping in prod).")
        full_train_data = lgb.Dataset(X_train_all, label=y_train_all)
        final_model = lgb.train(best_params, full_train_data, num_boost_round=num_boost_round)
        logger.info("Production model training complete.")


    else: # --- Normal Mode (Train/Test with Tuning) ---
        logger.info("--- Running in NORMAL MODE ---")
        # Initial Train/Test Split by Season (Identical)
        train_seasons=StrikeoutModelConfig.DEFAULT_TRAIN_YEARS; test_seasons=StrikeoutModelConfig.DEFAULT_TEST_YEARS
        logger.info(f"Train seasons {train_seasons}, Test seasons {test_seasons}")
        train_df=df[df['season'].isin(train_seasons)].copy().reset_index(drop=True); test_df=df[df['season'].isin(test_seasons)].copy().reset_index(drop=True)
        if train_df.empty or test_df.empty: logger.error(f"Train ({len(train_df)}) or Test ({len(test_df)}) empty."); return None, None
        logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
        X_train=train_df[feature_cols]; y_train=train_df['strikeouts']; X_test=test_df[feature_cols]; y_test=test_df['strikeouts']

        if best_params is None: # Run Optuna if no params provided
            logger.info("Running Optuna optimization using Time-Series CV for W1 Accuracy...")
            optuna_start_time = time.time()

            # --- Time Series Cross-Validation Setup ---
            cv_train_dfs = []
            cv_val_dfs = []
            # Diagnostic logging
            logger.info(f"Data type of 'season' column in train_df: {train_df['season'].dtype}")
            unique_seasons_in_train = sorted(train_df['season'].unique())
            logger.info(f"Unique seasons found in train_df: {unique_seasons_in_train}")
            logger.info(f"Value counts for 'season' in train_df:\n{train_df['season'].value_counts().sort_index()}")

            train_seasons_sorted = unique_seasons_in_train

            if len(train_seasons_sorted) < 2:
                logger.warning(f"Not enough unique seasons ({len(train_seasons_sorted)}) in train_df for cross-validation. Skipping Optuna.")
                best_params = { # Define defaults if Optuna skipped
                     'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'n_jobs': -1,
                     'verbose': -1, 'seed': StrikeoutModelConfig.RANDOM_STATE, 'learning_rate': 0.05,
                     'num_leaves': 31, 'max_depth': -1, 'min_data_in_leaf': 20, 'feature_fraction': 0.8,
                     'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.01, 'reg_lambda': 0.01
                 }
                best_params_list = [best_params]
                cv_metrics = {'within_1': [-1.0]} # Store -1 W1 if skipped

            else:
                logger.info(f"Attempting to create CV folds from {len(train_seasons_sorted)} seasons...")
                for i in range(1, len(train_seasons_sorted)):
                    val_season = train_seasons_sorted[i]
                    train_cv_seasons = train_seasons_sorted[:i]
                    cv_train_df_fold = train_df[train_df['season'].isin(train_cv_seasons)].copy()
                    cv_val_df_fold = train_df[train_df['season'] == val_season].copy()
                    logger.debug(f"Attempting Fold {i}: Val Season={val_season}, Train Seasons={train_cv_seasons}. Train size={len(cv_train_df_fold)}, Val size={len(cv_val_df_fold)}")
                    if not cv_train_df_fold.empty and not cv_val_df_fold.empty:
                        cv_train_dfs.append(cv_train_df_fold)
                        cv_val_dfs.append(cv_val_df_fold)
                        logger.info(f"Successfully created CV fold {i}: Training on {train_cv_seasons} ({len(cv_train_df_fold)} rows), validating on {val_season} ({len(cv_val_df_fold)} rows)")
                    else:
                        logger.warning(f"Skipping CV fold {i}: Empty train ({len(cv_train_df_fold)}) or validation ({len(cv_val_df_fold)}) set.")

                if not cv_train_dfs:
                    logger.error("No valid CV folds created after checking all potential splits. Cannot perform Optuna optimization.")
                    best_params = { # Define defaults if no folds created
                         'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'n_jobs': -1,
                         'verbose': -1, 'seed': StrikeoutModelConfig.RANDOM_STATE, 'learning_rate': 0.05,
                         'num_leaves': 31, 'max_depth': -1, 'min_data_in_leaf': 20, 'feature_fraction': 0.8,
                         'bagging_fraction': 0.8, 'bagging_freq': 5, 'reg_alpha': 0.01, 'reg_lambda': 0.01
                     }
                    best_params_list = [best_params]
                    cv_metrics = {'within_1': [-1.0]}

                else: # Proceed with Optuna using the created folds
                    logger.info(f"Successfully created {len(cv_train_dfs)} CV folds. Proceeding with Optuna.")
                    best_params_list = []
                    cv_metrics = {'within_1': []} # Store best W1 for each fold

                    for i, (cv_train_df, cv_val_df) in enumerate(zip(cv_train_dfs, cv_val_dfs)):
                        fold_start_time = time.time()
                        fold_num = i + 1 # Use actual fold number
                        val_season = cv_val_df['season'].unique()[0]
                        logger.info(f"Optimizing LightGBM fold {fold_num} for W1 Accuracy (Validate on Season {val_season})...")

                        X_train_cv = cv_train_df[feature_cols]
                        y_train_cv = cv_train_df['strikeouts']
                        X_val_cv = cv_val_df[feature_cols]
                        y_val_cv = cv_val_df['strikeouts']

                        tr_data = lgb.Dataset(X_train_cv, label=y_train_cv)
                        v_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=tr_data)

                        # Define the objective function for Optuna targeting W1 Accuracy
                        def objective_fold_lgbm_w1(trial):
                            params = {
                                'objective': 'regression', # Use RMSE objective for training
                                'metric': 'rmse',          # Use RMSE metric for early stopping
                                'boosting_type': 'gbdt',
                                'feature_pre_filter': False, # Keep this fix
                                'n_jobs': -1, 'verbose': -1,
                                'seed': StrikeoutModelConfig.RANDOM_STATE,
                                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                                'max_depth': trial.suggest_int('max_depth', 3, 12),
                                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True), # L1
                                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), # L2
                            }
                            model = lgb.train(params, tr_data, num_boost_round=2000,
                                              valid_sets=[v_data], valid_names=['validation'],
                                              callbacks=[lgb.early_stopping(100, verbose=False)]) # Early stopping based on rmse
                            preds = model.predict(X_val_cv, num_iteration=model.best_iteration)
                            w1_accuracy = within_n_strikeouts(y_val_cv, preds, n=1) # Calculate W1
                            return w1_accuracy # Return W1 score for Optuna to maximize

                        # Run Optuna study - Maximize W1 Accuracy
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective_fold_lgbm_w1, n_trials=100, timeout=None) # 100 trials

                        fold_best_params = study.best_params
                        fold_best_score = study.best_value
                        best_params_list.append(fold_best_params)
                        cv_metrics['within_1'].append(fold_best_score) # Store W1

                        logger.info(f"Fold {fold_num} best score (W1 Accuracy): {fold_best_score:.4f}")
                        logger.info(f"Fold {fold_num} best parameters: {fold_best_params}")
                        logger.info(f"Fold {fold_num} completed in {(time.time() - fold_start_time)/60:.1f} minutes.")

            # --- Select Best Parameters overall from CV ---
            if cv_metrics['within_1'] and not all(m == -1.0 for m in cv_metrics['within_1']): # Check if any folds ran
                 best_fold_idx = np.argmax(cv_metrics['within_1']) # Find index of maximum W1
                 best_params = best_params_list[best_fold_idx]
                 logger.info(f"\nSelected best parameters from fold {best_fold_idx + 1} (W1 Accuracy: {cv_metrics['within_1'][best_fold_idx]:.4f})")
            elif 'best_params' not in locals() or best_params is None:
                 logger.error("Optuna tuning failed and no default parameters available. Exiting.")
                 return None, None
            # If Optuna was skipped or failed but defaults exist, best_params is already set

            logger.info(f"Optuna total time: {(time.time() - optuna_start_time)/60:.1f} minutes.")

        else: # Use provided hyperparams
            logger.info(f"Using provided hyperparameters: {best_params}")


        # --- Train final model for normal mode (using user's early stopping strategy) ---
        logger.info("Training final model on full training set (using early stopping)...")
        # Ensure base params present if loaded from file or defaults used
        best_params.setdefault('objective','regression'); best_params.setdefault('metric','rmse'); # Align with objective func choice
        best_params.setdefault('verbose',-1); best_params.setdefault('n_jobs',-1);
        best_params.setdefault('seed', StrikeoutModelConfig.RANDOM_STATE)
        best_params.setdefault('feature_pre_filter', False)

        # Split training data again for final early stopping (using user's method)
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_train, y_train, test_size=0.15, random_state=StrikeoutModelConfig.RANDOM_STATE, shuffle=False
        )
        lgb_train_final = lgb.Dataset(X_train_final, label=y_train_final)
        lgb_val_final = lgb.Dataset(X_val_final, label=y_val_final, reference=lgb_train_final)

        logger.info("Training with early stopping based on validation RMSE/MAE (depends on metric param)...")
        final_model = lgb.train(
            best_params,
            lgb_train_final,
            num_boost_round=2500, # High limit
            valid_sets=[lgb_val_final],
            valid_names=['early_stop_validation'],
            # Early stopping will use the 'metric' defined in best_params (e.g., rmse or mae)
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)]
        )
        logger.info(f"Normal mode training complete. Best iteration: {final_model.best_iteration}")

        # --- Evaluation (Identical) ---
        logger.info(f"Evaluating on Test Set ({len(test_df)} rows)...")
        test_preds = final_model.predict(X_test, num_iteration=final_model.best_iteration)
        final_rmse=np.sqrt(mean_squared_error(y_test,test_preds)); final_mae=mean_absolute_error(y_test,test_preds)
        final_w1=within_n_strikeouts(y_test,test_preds,n=1); final_w2=within_n_strikeouts(y_test,test_preds,n=2)
        logger.info(f"--- Test Metrics ---"); logger.info(f"RMSE:{final_rmse:.4f} MAE:{final_mae:.4f} W1:{final_w1:.4f} W2:{final_w2:.4f}")

        # --- Save Plots & Importance CSV (Identical logic, ensures CSV is saved) ---
        output_dir = project_root / 'models'; plots_dir = project_root / 'plots'; plots_dir.mkdir(exist_ok=True, parents=True); timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "prod" if production_mode else "test"
        try: # Feat Imp
            imp_df=pd.DataFrame({'Feature':feature_cols,'Importance':final_model.feature_importance()}).sort_values(by='Importance',ascending=False)
            full_importance_filename = output_dir / f'{prefix}_feature_importance_full_{timestamp}.csv' # Save full list
            imp_df.to_csv(full_importance_filename, index=False); logger.info(f"Full importance list saved: {full_importance_filename}")
            plt.figure(figsize=(10,max(6,len(feature_cols)//4))); top_n=min(30,len(feature_cols)); sns.barplot(x='Importance',y='Feature',data=imp_df.head(top_n)); plt.title(f'Top {top_n} Feats ({timestamp})'); plt.tight_layout(); p=plots_dir/f'{prefix}_feat_imp_{timestamp}.png'; plt.savefig(p); plt.close(); logger.info(f"Importance plot: {p}")
        except Exception as e: logger.error(f"Importance processing/plot fail: {e}")
        try: # Pred vs Actual
            plt.figure(figsize=(8,8)); plt.scatter(y_test,test_preds,alpha=0.4,label='P'); plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--',label='Perf'); plt.xlabel('Actual'); plt.ylabel('Pred'); plt.title(f'Pred vs Actual ({timestamp})'); plt.legend(); plt.grid(True); plt.tight_layout(); p=plots_dir/f'{prefix}_pred_actual_{timestamp}.png'; plt.savefig(p); plt.close(); logger.info(f"Pred/Actual plot: {p}")
        except Exception as e: logger.error(f"Pred/Actual plot fail: {e}")


    # --- Save Model & Features (Identical logic) ---
    if final_model:
        output_dir = project_root / 'models'; output_dir.mkdir(exist_ok=True)
        prefix = "prod" if production_mode else "test"; timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Redefine if needed
        model_file = output_dir / f'{prefix}_strikeout_model_{timestamp}.txt'; final_model.save_model(str(model_file)); logger.info(f"Model saved: {model_file}")
        if best_params and not hyperparams: # Save if Optuna ran
            params_file = output_dir / f'{prefix}_best_params_{timestamp}.json'
            try:
                # Convert numpy types if necessary for JSON
                serializable_params = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in best_params.items()}
                with open(params_file, 'w') as f: json.dump(serializable_params, f, indent=4); logger.info(f"Best params saved: {params_file}")
            except Exception as e: logger.error(f"Failed save best params: {e}")

        features_file = output_dir / f'{prefix}_feature_columns_{timestamp}.pkl'
        try:
            with open(features_file, 'wb') as f: pickle.dump(feature_cols, f); logger.info(f"Features saved: {features_file}")
        except Exception as e: logger.error(f"Failed save features: {e}")

    return final_model, feature_cols

# --- Argument Parser and Main Execution (Identical) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM Strikeout Prediction Model.")
    parser.add_argument("--production-mode", action="store_true", help="Train final model on all historical data.")
    parser.add_argument("--params-file", type=str, help="Path to JSON file with hyperparameters (skips Optuna).")
    return parser.parse_args()

if __name__ == "__main__":
    if not MODULE_IMPORTS_OK: sys.exit("Exiting: Failed module imports.")
    args = parse_args(); params = None
    if args.params_file:
        try:
            with open(args.params_file, 'r') as f: params = json.load(f); logger.info(f"Loaded params from {args.params_file}")
        except Exception as e: logger.error(f"Failed load params from {args.params_file}: {e}"); sys.exit(1)
    logger.info(f"Starting training (Production Mode: {args.production_mode})...")
    model, features = train_model(production_mode=args.production_mode, hyperparams=params)
    if model and features: logger.info("Model training finished successfully."); sys.exit(0)
    else: logger.error("Model training failed."); sys.exit(1)