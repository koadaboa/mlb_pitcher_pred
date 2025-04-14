# src/models/train_complete_model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna # Keep Optuna for standard training/testing mode
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Import train_test_split for final training validation
from sklearn.model_selection import train_test_split
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
    def setup_logger(name, level=logging.INFO, log_file=None): logging.basicConfig(level=level); return logging.getLogger(name)
    class DBConnection:
        def __init__(self, p): self.p=p; print(f"WARN: Using dummy DBConnection for {p}")
        def __enter__(self): return None # Allow basic execution
        def __exit__(self,t,v,tb): pass
    class StrikeoutModelConfig: RANDOM_STATE=42; DEFAULT_TRAIN_YEARS=(); DEFAULT_TEST_YEARS=(); WINDOW_SIZES=()
    class DBConfig: PATH="data/pitcher_stats.db"

# Setup logger
log_dir = project_root / 'logs'
log_dir.mkdir(exist_ok=True, parents=True)
logger = setup_logger('train_complete_model', log_file= log_dir / 'train_model.log', level=logging.INFO) if MODULE_IMPORTS_OK else logging.getLogger('train_fallback')

# --- Helper Function ---
def within_n_strikeouts(y_true, y_pred, n=1):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    within_n = np.abs(y_true - np.round(y_pred)) <= n
    return np.mean(within_n)

# --- Main Training Function ---
def train_model(production_mode=False, hyperparams=None):
    db_path = project_root / DBConfig.PATH
    logger.info("Loading combined predictive feature data...")
    all_historical_data = []
    try:
        with DBConnection(db_path) as conn:
             if conn is None: raise ConnectionError("DB Connection failed.")
             # Load train/test features
             try:
                  train_df=pd.read_sql_query("SELECT * FROM train_combined_features", conn); logger.info(f"Loaded {len(train_df)} train rows")
                  if not train_df.empty: all_historical_data.append(train_df)
             except Exception as e: logger.warning(f"No train_combined_features: {e}")
             try:
                  test_df=pd.read_sql_query("SELECT * FROM test_combined_features", conn); logger.info(f"Loaded {len(test_df)} test rows")
                  if not test_df.empty: all_historical_data.append(test_df)
             except Exception as e: logger.warning(f"No test_combined_features: {e}")
        if not all_historical_data: logger.error("No historical data loaded."); return None, None
        df = pd.concat(all_historical_data, ignore_index=True); logger.info(f"Total rows: {len(df)}")
    except Exception as e: logger.error(f"Failed load features: {e}", exc_info=True); return None, None

    # --- Data Prep ---
    df['game_date'] = pd.to_datetime(df['game_date']).dt.date; df = df.sort_values(['pitcher_id', 'game_date'])
    numeric_cols = df.select_dtypes(include=np.number).columns; imputation_values = {}; logger.info("Imputing missing values...")
    for col in numeric_cols:
        if df[col].isnull().any(): median_val = df[col].median(); df[col]=df[col].fillna(median_val); imputation_values[col]=median_val

    # --- Feature Selection ---
    exclude_cols = [ # Use same consistent list
        'index', '', 'pitcher_id', 'player_name', 'game_pk', 'home_team', 'away_team',
        'opponent_team_name', 'game_date', 'season', 'game_month', 'year',
        'p_throws', 'stand', 'team', 'Team', 'opp_base_team', 'opp_adv_team', 'opp_adv_opponent',
        'strikeouts', # Target
        'k_per_9', 'k_percent', 'batters_faced', 'total_pitches', 'avg_velocity',
        'max_velocity', 'avg_spin_rate', 'avg_horizontal_break', 'avg_vertical_break',
        'zone_percent', 'swinging_strike_percent', 'innings_pitched', 'fastball_percent',
        'breaking_percent', 'offspeed_percent', 'inning', 'score_differential',
        'is_close_game', 'is_playoff'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.int64, np.float64, 'int', 'float', 'int32', 'float32', 'Int64']]
    logger.info(f"Using {len(feature_cols)} features.")
    if len(feature_cols) < 5: logger.warning(f"Low feature count: {feature_cols}.")

    # --- Mode Logic ---
    final_model = None; best_params = hyperparams

    if production_mode:
        # --- Production Mode ---
        logger.info("--- Running in PRODUCTION MODE ---")
        X_train_all = df[feature_cols]; y_train_all = df['strikeouts']
        if X_train_all.empty: logger.error("No training data."); return None, None
        logger.info(f"Training production model on {len(X_train_all)} samples.")
        if best_params is None:
             logger.warning("No hyperparameters provided. Using defaults.")
             best_params = {'objective':'regression_l1', 'metric':'mae', 'boosting_type':'gbdt', 'learning_rate': 0.05, 'num_leaves': 50, 'max_depth': 7, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'n_jobs': -1, 'seed': StrikeoutModelConfig.RANDOM_STATE}
        else: logger.info(f"Using provided hyperparameters: {best_params}")
        # Ensure base params present
        best_params.setdefault('objective','regression_l1'); best_params.setdefault('metric','mae'); best_params.setdefault('verbose',-1); best_params.setdefault('n_jobs',-1);
        num_boost_round = best_params.pop('num_boost_round', 1500) # Use a fixed large number or from params
        logger.info(f"Training for max {num_boost_round} rounds (no early stopping in prod).")
        full_train_data = lgb.Dataset(X_train_all, label=y_train_all)
        # Train without validation/early stopping for production model
        final_model = lgb.train(best_params, full_train_data, num_boost_round=num_boost_round)
        logger.info("Production model training complete.")

    else: # --- Normal Mode ---
        logger.info("--- Running in NORMAL MODE ---")
        train_seasons=StrikeoutModelConfig.DEFAULT_TRAIN_YEARS; test_seasons=StrikeoutModelConfig.DEFAULT_TEST_YEARS
        logger.info(f"Train seasons {train_seasons}, Test seasons {test_seasons}")
        train_df=df[df['season'].isin(train_seasons)].copy().reset_index(drop=True); test_df=df[df['season'].isin(test_seasons)].copy().reset_index(drop=True)
        if train_df.empty or test_df.empty: logger.error(f"Train ({len(train_df)}) or Test ({len(test_df)}) empty."); return None, None
        logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
        X_train=train_df[feature_cols]; y_train=train_df['strikeouts']; X_test=test_df[feature_cols]; y_test=test_df['strikeouts']

        if best_params is None: # Run Optuna if no params provided
             logger.info("Running Optuna optimization..."); optuna_start_time = time.time()
             def objective(trial):
                  params={'objective':'regression_l1','metric':'mae','boosting_type':'gbdt','n_jobs':-1,'verbose':-1,'seed':StrikeoutModelConfig.RANDOM_STATE,'learning_rate':trial.suggest_float('learning_rate',0.01,0.1,log=True),'num_leaves':trial.suggest_int('num_leaves',20,100),'max_depth':trial.suggest_int('max_depth',3,10),'min_data_in_leaf':trial.suggest_int('min_data_in_leaf',20,100),'feature_fraction':trial.suggest_float('feature_fraction',0.6,1.0),'bagging_fraction':trial.suggest_float('bagging_fraction',0.6,1.0),'bagging_freq':trial.suggest_int('bagging_freq',1,7),'reg_alpha':trial.suggest_float('reg_alpha',1e-3,10.0,log=True),'reg_lambda':trial.suggest_float('reg_lambda',1e-3,10.0,log=True)}
                  # Split train data further for Optuna validation
                  X_tr_opt, X_v_opt, y_tr_opt, y_v_opt = train_test_split(X_train,y_train,test_size=0.2,random_state=StrikeoutModelConfig.RANDOM_STATE,shuffle=False)
                  tr_data=lgb.Dataset(X_tr_opt,label=y_tr_opt); v_data=lgb.Dataset(X_v_opt,label=y_v_opt,reference=tr_data)
                  model=lgb.train(params,tr_data,num_boost_round=1500, # Higher max rounds for Optuna
                                   valid_sets=[v_data],valid_names=['validation'],callbacks=[lgb.early_stopping(100,verbose=False)]) # Increased patience
                  preds=model.predict(X_v_opt,num_iteration=model.best_iteration); mae=mean_absolute_error(y_v_opt,preds); return mae # Return MAE to minimize
             study = optuna.create_study(direction='minimize')
             study.optimize(objective, n_trials=50, timeout=1800) # Adjust trials/timeout as needed
             best_params = study.best_params # Get best param combination
             # !! REMOVED line trying to access study.best_trial.best_iteration !!
             logger.info(f"Optuna finished in {(time.time() - optuna_start_time)/60:.1f} minutes. Best MAE: {study.best_value:.4f}")
             logger.info(f"Best parameters found: {best_params}")
             # Note: We don't get num_boost_round from Optuna here anymore
        else: logger.info(f"Using provided hyperparameters: {best_params}")

        # --- Train final model for normal mode ---
        logger.info("Training final model on full training set (using early stopping)...")
        best_params.setdefault('objective','regression_l1'); best_params.setdefault('metric','mae'); best_params.setdefault('verbose',-1); best_params.setdefault('n_jobs',-1);
        # --- MODIFICATION: Use early stopping for final training ---
        # Split training data again to get a validation set for early stopping
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
             X_train, y_train, test_size=0.15, random_state=StrikeoutModelConfig.RANDOM_STATE, shuffle=False # Smaller validation set
        )
        lgb_train_final = lgb.Dataset(X_train_final, label=y_train_final)
        lgb_val_final = lgb.Dataset(X_val_final, label=y_val_final, reference=lgb_train_final)

        logger.info("Training with early stopping...")
        final_model = lgb.train(
             best_params,
             lgb_train_final,
             num_boost_round=2500, # Set a high limit for rounds
             valid_sets=[lgb_val_final],
             valid_names=['early_stop_validation'],
             callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)] # Stop if val MAE doesn't improve
        )
        logger.info(f"Normal mode training complete. Best iteration: {final_model.best_iteration}")
        # --- END MODIFICATION ---

        # --- Evaluation ---
        logger.info(f"Evaluating on Test Set ({len(test_df)} rows)...")
        test_preds = final_model.predict(X_test, num_iteration=final_model.best_iteration) # Use best iteration
        final_rmse=np.sqrt(mean_squared_error(y_test,test_preds)); final_mae=mean_absolute_error(y_test,test_preds)
        final_w1=within_n_strikeouts(y_test,test_preds,n=1); final_w2=within_n_strikeouts(y_test,test_preds,n=2)
        logger.info(f"--- Test Metrics ---"); logger.info(f"RMSE:{final_rmse:.4f} MAE:{final_mae:.4f} W1:{final_w1:.4f} W2:{final_w2:.4f}")

        # --- Save Plots --- (Identical plotting code as before)
        output_dir = project_root / 'models'; plots_dir = project_root / 'plots'; plots_dir.mkdir(exist_ok=True); timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try: # Feat Imp
             imp_df=pd.DataFrame({'F':feature_cols,'I':final_model.feature_importance()}).sort_values(by='I',ascending=False); plt.figure(figsize=(10,max(6,len(feature_cols)//2))); top_n=min(30,len(feature_cols)); sns.barplot(x='I',y='F',data=imp_df.head(top_n)); plt.title(f'Top {top_n} Feats ({timestamp})'); plt.tight_layout(); p=plots_dir/f'test_feat_imp_{timestamp}.png'; plt.savefig(p); plt.close(); logger.info(f"Importance plot: {p}")
        except Exception as e: logger.error(f"Importance plot fail: {e}")
        try: # Pred vs Actual
             plt.figure(figsize=(8,8)); plt.scatter(y_test,test_preds,alpha=0.4,label='P'); plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--',label='Perf'); plt.xlabel('Actual'); plt.ylabel('Pred'); plt.title(f'Pred vs Actual ({timestamp})'); plt.legend(); plt.grid(True); plt.tight_layout(); p=plots_dir/f'test_pred_actual_{timestamp}.png'; plt.savefig(p); plt.close(); logger.info(f"Pred/Actual plot: {p}")
        except Exception as e: logger.error(f"Pred/Actual plot fail: {e}")


    # --- Save Model & Features ---
    if final_model:
        output_dir = project_root / 'models'; output_dir.mkdir(exist_ok=True)
        prefix = "prod" if production_mode else "test"; timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = output_dir / f'{prefix}_strikeout_model_{timestamp}.txt'; final_model.save_model(str(model_file)); logger.info(f"Model saved: {model_file}")
        features_file = output_dir / f'{prefix}_feature_columns_{timestamp}.pkl'
        try:
            with open(features_file, 'wb') as f: pickle.dump(feature_cols, f); logger.info(f"Features saved: {features_file}")
        except Exception as e: logger.error(f"Failed save features: {e}")

    return final_model, feature_cols

# --- Argument Parser and Main Execution (Identical) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM Strikeout Prediction Model.")
    parser.add_argument("--production-mode", action="store_true", help="Train final model on all historical data.")
    parser.add_argument("--params-file", type=str, help="Path to JSON file with hyperparameters.")
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