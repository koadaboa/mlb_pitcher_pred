# src/models/train_complete_model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import os
import sys
import pickle
from pathlib import Path

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection
from src.config import StrikeoutModelConfig

# Setup logger
logger = setup_logger('train_complete_model')

def within_n_strikeouts(y_true, y_pred, n=1):
    """Calculate percentage of predictions within n strikeouts of actual value"""
    # Ensure y_true and y_pred are numpy arrays for vectorized operations
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    within_n = np.abs(y_true - np.round(y_pred)) <= n
    return np.mean(within_n)

# Removed the unused load_data function as train_model loads the combined data directly.

def train_model():
    """
    Train LightGBM model with combined pitcher, batter, and team features,
    ensuring no data leakage through proper time-series splitting and feature exclusion.
    """
    # Load combined feature data (created by engineer_features.py)
    # This table includes pitcher features, batter features, and relevant team features.
    logger.info("Loading combined predictive feature data...")
    with DBConnection() as conn:
        # Load BOTH train and test features
        train_query = "SELECT * FROM train_combined_features"
        test_query = "SELECT * FROM test_combined_features"
        
        try:
            train_features_df = pd.read_sql_query(train_query, conn)
            logger.info(f"Loaded {len(train_features_df)} rows from train_combined_features")
            
            test_features_df = pd.read_sql_query(test_query, conn)
            logger.info(f"Loaded {len(test_features_df)} rows from test_combined_features")
            
            # Concatenate them into one DataFrame
            df = pd.concat([train_features_df, test_features_df], ignore_index=True)
            
        except Exception as e:
            logger.error(f"Failed to load train/test combined features: {e}")
            # Re-raise or handle appropriately
            raise pandas.errors.DatabaseError(f"Execution failed on loading train/test combined features: {e}") from e

    if df.empty:
        logger.error("Failed to load combined features. Make sure engineer_features.py ran successfully.")
        return None, None

    # --- Data Preparation ---
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values(['pitcher_id', 'game_date']) # Sort for time series integrity

    # Handle missing values (using median imputation)
    for col in df.select_dtypes(include=np.number).columns: # Use np.number for broader numeric types
        if df[col].isnull().any(): # More efficient check
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled {df[col].isnull().sum()} missing values in {col} with median ({median_val})")

    # --- Feature Selection (Preventing Leakage) ---
    # Exclude target variable ('strikeouts'), identifiers, date/season info used for splitting,
    # and any features derived *from the current game's events* (potential leaks).
    # Features like rolling stats, career stats, etc., should have been calculated
    # using *shifted* data in the feature engineering step to represent past performance.
    exclude_cols = [
        'index', '', # Generic index columns if they exist
        'pitcher_id', 'player_name', 'game_pk', # Identifiers
        'home_team', 'away_team', 'opposing_team', 'player_name_opp', # Team/Opponent Names (usually represented by features)
        'game_date', 'season', 'game_month', # Date/Time related (Season is used for splitting)
        'p_throws', # Pitcher handedness (might be useful, but exclude if redundant with features)

        # --- TARGET VARIABLE ---
        'strikeouts',

        # --- POTENTIAL LEAKAGE COLUMNS (derived from CURRENT game performance) ---
        # These should represent *past* performance (e.g., rolling_5g_k_pct)
        # not the performance *in the game being predicted*.
        'k_per_9',                  # Current game K/9
        'k_percent',                # Current game K%
        'batters_faced',            # Current game BF
        'total_pitches',            # Current game Pitches
        'avg_velocity',             # Current game Avg Velo
        'max_velocity',             # Current game Max Velo
        'avg_spin_rate',            # Current game Avg Spin
        'avg_horizontal_break',     # Current game Avg Horiz Break
        'avg_vertical_break',       # Current game Avg Vert Break
        'zone_percent',             # Current game Zone%
        'swinging_strike_percent',  # Current game SwStr% - HIGH LEAKAGE RISK
        'innings_pitched',          # Current game IP
        'fastball_percent',         # Current game Fastball %
        'breaking_percent',         # Current game Breaking %
        'offspeed_percent',         # Current game Offspeed %
        # Add any other columns calculated *during* the game being predicted
        'is_playoff',
        'is_home',
        'is_close_game',
        'score_differential',
        'inning',
        'rest_days_6_more',
        'rest_days_4_less',
        'is_month_3',
        'is_month_5',
        'is_month_6',
        'is_month_7',
        'is_month_8'
    ]
    # Keep only columns that are not in the exclude list
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.int64, np.float64, 'int', 'float']] # Ensure numeric types
    logger.info(f"Using {len(feature_cols)} features in complete model: {feature_cols[:5]}...") # Log first few features

    # --- Strict Time-Based Train/Test Split ---
    # Split data based on predefined seasons *before* any training or optimization.
    # This is crucial to prevent data leakage.
    train_seasons = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
    test_seasons = StrikeoutModelConfig.DEFAULT_TEST_YEARS

    logger.info(f"Training on seasons {train_seasons}, testing on {test_seasons}")

    train_df = df[df['season'].isin(train_seasons)].copy().reset_index(drop=True)
    test_df = df[df['season'].isin(test_seasons)].copy().reset_index(drop=True)

    if train_df.empty or test_df.empty:
        logger.error(f"Train or test set is empty after season split. Train: {len(train_df)}, Test: {len(test_df)}")
        return None, None
    logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")

    # --- Time Series Cross-Validation and Hyperparameter Optimization (Optuna) ---
    # Use time series CV *within the training data only* to find best parameters.
    cv_train_dfs = []
    cv_val_dfs = []
    train_seasons_sorted = sorted(train_df['season'].unique())

    # Create CV folds: train on earlier seasons, validate on the next season
    if len(train_seasons_sorted) < 2:
        logger.warning("Not enough seasons in the training data for cross-validation. Skipping Optuna.")
        # Use default parameters or train without CV optimization
        best_params = { # Example default parameters
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
            'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1,
             'min_data_in_leaf': 20, 'feature_fraction': 0.9, 'bagging_fraction': 0.8,
             'bagging_freq': 5, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'verbose': -1
        }
        best_params_list = [best_params] # Store defaults
    else:
        for i in range(1, len(train_seasons_sorted)):
            val_season = train_seasons_sorted[i]
            train_cv_seasons = train_seasons_sorted[:i]

            cv_train_df = train_df[train_df['season'].isin(train_cv_seasons)].copy()
            cv_val_df = train_df[train_df['season'] == val_season].copy()

            if cv_train_df.empty or cv_val_df.empty:
                 logger.warning(f"Skipping CV fold {i}: Empty train or validation set for season {val_season}")
                 continue

            cv_train_dfs.append(cv_train_df)
            cv_val_dfs.append(cv_val_df)
            logger.info(f"CV fold {i}: Training on {train_cv_seasons}, validating on {val_season}")

        best_params_list = []
        cv_metrics = {'within_1': []} # Store validation metrics per fold

        if not cv_train_dfs:
             logger.error("No valid CV folds created. Cannot perform Optuna optimization.")
             # Fallback to default parameters
             best_params = { # Example default parameters
                'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
                'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': -1,
                'min_data_in_leaf': 20, 'feature_fraction': 0.9, 'bagging_fraction': 0.8,
                'bagging_freq': 5, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'verbose': -1
             }
             best_params_list = [best_params]
        else:
            # Optuna optimization loop for each CV fold
            for i, (cv_train_df, cv_val_df) in enumerate(zip(cv_train_dfs, cv_val_dfs)):
                fold_start_time = datetime.now()
                val_season = cv_val_df['season'].unique()[0]
                logger.info(f"Optimizing fold {i+1} (Validate on Season {val_season})...")

                # Define the objective function for Optuna for this fold
                def objective_fold(trial):
                    params = {
                        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                        'verbose': -1, 'n_jobs': -1 # Use all available cores
                    }

                    X_train_cv = cv_train_df[feature_cols]
                    y_train_cv = cv_train_df['strikeouts']
                    X_val_cv = cv_val_df[feature_cols]
                    y_val_cv = cv_val_df['strikeouts']

                    train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
                    val_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data)

                    # Use a reasonable number of boosting rounds with early stopping
                    model = lgb.train(
                        params, train_data, num_boost_round=2000, # Increased rounds
                        valid_sets=[val_data], valid_names=['validation'],
                        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)] # Increased patience
                    )
                    y_pred_val = model.predict(X_val_cv)
                    within_1 = within_n_strikeouts(y_val_cv, y_pred_val, n=1)
                    return within_1 # Optuna maximizes this

                # Run Optuna study
                study = optuna.create_study(direction='maximize')
                # Increase n_trials for better exploration
                study.optimize(objective_fold, n_trials=75, timeout=600) # Add timeout

                fold_best_params = study.best_params
                fold_best_score = study.best_value
                best_params_list.append(fold_best_params)
                cv_metrics['within_1'].append(fold_best_score)

                logger.info(f"Fold {i+1} best score (within_1): {fold_best_score:.4f}")
                logger.info(f"Fold {i+1} best parameters: {fold_best_params}")
                logger.info(f"Fold {i+1} completed in {datetime.now() - fold_start_time}")

    # --- Final Model Training ---
    # Select best parameters (e.g., from the fold with the best validation score)
    if cv_metrics['within_1']:
         best_fold_idx = np.argmax(cv_metrics['within_1'])
         best_params = best_params_list[best_fold_idx]
         logger.info(f"\nSelected best parameters from fold {best_fold_idx + 1} (score: {cv_metrics['within_1'][best_fold_idx]:.4f})")
    elif best_params_list: # Handle case where CV didn't run but defaults exist
         best_params = best_params_list[0]
         logger.info("\nUsing default parameters as CV did not run or produced no results.")
    else:
         logger.error("No suitable parameters found. Exiting.")
         return None, None


    # Add fixed parameters
    best_params['objective'] = 'regression'
    best_params['metric'] = 'rmse' # Use RMSE for training loss, but optimized for 'within_1'
    best_params['verbose'] = -1
    best_params['n_jobs'] = -1

    logger.info(f"\nTraining final model on all training data ({len(train_df)} rows) with parameters: {best_params}")

    X_train_all = train_df[feature_cols]
    y_train_all = train_df['strikeouts']
    full_train_data = lgb.Dataset(X_train_all, label=y_train_all)

    # Train the final model - potentially increase boosting rounds and use early stopping
    # based on training data if no separate validation set is held out from training.
    # Or train for a fixed number of rounds determined during CV.
    # Let's find the optimal boosting rounds using CV results if available.
    # This part requires careful consideration based on the CV setup.
    # For simplicity here, train for a fixed, reasonably large number of rounds.
    final_model = lgb.train(
        best_params,
        full_train_data,
        num_boost_round=1500 # Adjust as needed, potentially based on CV optimal rounds
        # No validation set here as we use the entire training data
    )

    # --- Final Evaluation on Separate Test Set ---
    logger.info(f"\nEvaluating final model on separate test set ({len(test_df)} rows)...")
    X_test = test_df[feature_cols]
    y_test = test_df['strikeouts']

    final_preds = final_model.predict(X_test)

    # Calculate final metrics
    final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    final_mae = mean_absolute_error(y_test, final_preds)
    final_within_1 = within_n_strikeouts(y_test, final_preds, n=1)
    final_within_2 = within_n_strikeouts(y_test, final_preds, n=2)

    logger.info(f"--- Final Test Set Metrics ---")
    logger.info(f"RMSE: {final_rmse:.4f}")
    logger.info(f"MAE: {final_mae:.4f}")
    logger.info(f"Within 1 Strikeout: {final_within_1:.4f}")
    logger.info(f"Within 2 Strikeouts: {final_within_2:.4f}")

    # --- Save Artifacts ---
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True, parents=True)
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': final_model.feature_importance()
    }).sort_values(by='Importance', ascending=False)

    full_importance_filename = output_dir / f'complete_feature_importance_full_{timestamp}.csv'
    try:
        importance_df.to_csv(full_importance_filename, index=False)
        logger.info(f"Full feature importance list saved to {full_importance_filename}")
    except Exception as e:
        logger.error(f"Failed to save full feature importance CSV: {e}")

    plt.figure(figsize=(10, 12)) # Adjusted size
    top_n = 30 # Show top N features
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
    plt.title(f'Top {top_n} Feature Importance - Final Model ({timestamp})')
    plt.tight_layout()
    importance_plot_path = plots_dir / f'complete_feature_importance_{timestamp}.png'
    plt.savefig(importance_plot_path)
    logger.info(f"Feature importance plot saved to {importance_plot_path}")
    plt.close() # Close plot

    # Predictions vs Actuals Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, final_preds, alpha=0.4, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Strikeouts')
    plt.ylabel('Predicted Strikeouts')
    plt.title(f'Predicted vs Actual Strikeouts - Test Set ({timestamp})')
    plt.legend()
    plt.grid(True)
    pred_actual_plot_path = plots_dir / f'complete_pred_vs_actual_{timestamp}.png'
    plt.savefig(pred_actual_plot_path)
    logger.info(f"Prediction vs Actual plot saved to {pred_actual_plot_path}")
    plt.close() # Close plot

    # Save Model
    model_filename = output_dir / f'complete_strikeout_predictor_{timestamp}.txt'
    final_model.save_model(str(model_filename)) # Use string path for older versions if needed
    logger.info(f"Final model saved to {model_filename}")

    # Save Feature Columns list
    features_filename = output_dir / f'complete_feature_columns_{timestamp}.pkl'
    with open(features_filename, 'wb') as f:
        pickle.dump(feature_cols, f)
    logger.info(f"Feature columns saved to {features_filename}")

    return final_model, feature_cols

if __name__ == "__main__":
    logger.info("Starting strikeout model training with complete features...")
    model, features = train_model()
    if model and features:
         logger.info("Complete model training finished successfully.")
    else:
         logger.error("Complete model training failed.")