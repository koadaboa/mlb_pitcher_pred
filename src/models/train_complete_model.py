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
from config import StrikeoutModelConfig

# Setup logger
logger = setup_logger('train_complete_model')

def within_n_strikeouts(y_true, y_pred, n=1):
    """Calculate percentage of predictions within n strikeouts of actual value"""
    within_n = np.abs(y_true - np.round(y_pred)) <= n
    return np.mean(within_n)

def load_data():
    """Load train and test data separately"""
    logger.info("Loading train and test feature data...")
    logger.info("Loading TRAIN/TEST PITCHER feature data...")
    with DBConnection() as conn:
        train_query = "SELECT * FROM train_predictive_pitch_features" # Load pitcher features directly
        train_df = pd.read_sql_query(train_query, conn)
        
        test_query = "SELECT * FROM test_predictive_pitch_features" # Load pitcher features directly
        test_df = pd.read_sql_query(test_query, conn)
    
    logger.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
    return train_df, test_df

def prepare_data(df):
    """Prepare data for modeling"""
    # Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Sort by pitcher_id and game_date for time series integrity
    df = df.sort_values(['pitcher_id', 'game_date'])
    
    # Handle missing values
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].isnull().sum() > 0:
            logger.info(f"Filling {df[col].isnull().sum()} missing values in {col}")
            df[col] = df[col].fillna(df[col].median())
    
    # Define features and target
    exclude_cols = [
        'index', '', 'pitcher_id', 'player_name', 'game_date', 'game_pk', 
        'home_team', 'away_team', 'p_throws', 'season', 'strikeouts',
        'player_name_opp', 'opposing_team', 'game_month', 'k_per_9', 'k_percent'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features in complete model")

    return df, feature_cols

def time_series_cv_by_season(df):
    """Time series cross-validation by season"""
    seasons = sorted(df['season'].unique())
    logger.info(f"Seasons for CV: {seasons}")
    
    splits = []
    for i, test_season in enumerate(seasons[1:], 1):
        train_seasons = seasons[:i]
        train_idx = df[df['season'].isin(train_seasons)].index
        test_idx = df[df['season'] == test_season].index
        splits.append((train_idx, test_idx))
        
    return splits

def objective(trial, df, feature_cols, train_idx, val_idx):
    """Optuna objective function for hyperparameter optimization"""
    # Define hyperparameters to tune
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'verbose': -1
    }
    
    # Get training and validation data
    X_train = df.iloc[train_idx][feature_cols]
    y_train = df.iloc[train_idx]['strikeouts']
    X_val = df.iloc[val_idx][feature_cols]
    y_val = df.iloc[val_idx]['strikeouts']
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=10000,
        valid_sets=[val_data],
        valid_names=['validation'],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # We want to maximize the percentage of predictions within 1 strikeout
    within_1 = within_n_strikeouts(y_val, y_pred, n=1)
    
    return within_1  # Optuna tries to maximize this

def train_model():
    """Train LightGBM model with all features while preventing data leakage"""
    # Load data
    logger.info("Loading preprocessed feature data...")
    with DBConnection() as conn:
        query = "SELECT * FROM combined_predictive_features"
        df = pd.read_sql_query(query, conn)
    
    logger.info(f"Loaded {len(df)} rows of combined features")
    
    # Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Sort by pitcher_id and game_date for time series integrity
    df = df.sort_values(['pitcher_id', 'game_date'])
    
    # Handle missing values
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].isnull().sum() > 0:
            logger.info(f"Filling {df[col].isnull().sum()} missing values in {col}")
            df[col] = df[col].fillna(df[col].median())
    
    # Define features and target
    exclude_cols = [
        'index', '', 'pitcher_id', 'player_name', 'game_date', 'game_pk',
        'home_team', 'away_team', 'p_throws', 'season',
        # Target and previously excluded:
        'strikeouts',
        'k_per_9',
        'k_percent',
        # Add these potential leaky columns from the *current* game:
        'batters_faced',
        'total_pitches',
        'avg_velocity', # Current game avg velocity
        'max_velocity', # Current game max velocity
        'avg_spin_rate', # Current game avg spin
        'avg_horizontal_break', # Current game avg movement
        'avg_vertical_break', # Current game avg movement
        'zone_percent', # Current game zone %
        'swinging_strike_percent', # <-- Very likely leak
        'innings_pitched', # Current game IP
        'fastball_percent', # Current game pitch mix
        'breaking_percent', # Current game pitch mix
        'offspeed_percent', # Current game pitch mix
        'player_name_opp',
        'opposing_team',
        'game_month'
        # Ensure any other column derived purely from the current game's events is also here
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    logger.info(f"Using {len(feature_cols)} features in complete model")
    
    # Strict time-based split by season (not allowing any leakage)
    train_seasons = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
    test_seasons = StrikeoutModelConfig.DEFAULT_TEST_YEARS
    
    logger.info(f"Training on seasons {train_seasons}, testing on {test_seasons}")
    
    # Create train/test split by season
    train_df = df[df['season'].isin(train_seasons)].copy()
    test_df = df[df['season'].isin(test_seasons)].copy()
    
    # Reset indices to avoid out-of-bounds errors
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    logger.info(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")

    
    
    # Metrics storage
    metrics = {
        'rmse': [], 'mae': [], 
        'within_1': [], 'within_2': []
    }
    
    # Best parameters and models
    best_params_list = []
    models = []
    importance_dfs = []
    
    # Create time series CV splits within training data only
    cv_train_dfs = []
    cv_val_dfs = []
    
    # Use the earliest season for initial training, validate on subsequent training seasons
    train_seasons_sorted = sorted(train_df['season'].unique())
    for i in range(1, len(train_seasons_sorted)):
        val_season = train_seasons_sorted[i]
        train_seasons_subset = train_seasons_sorted[:i]
        
        # Create actual DataFrame subsets instead of just indices
        cv_train_df = train_df[train_df['season'].isin(train_seasons_subset)].copy()
        cv_val_df = train_df[train_df['season'] == val_season].copy()
        
        cv_train_dfs.append(cv_train_df)
        cv_val_dfs.append(cv_val_df)
        
        logger.info(f"CV fold {i}: Training on {train_seasons_subset}, validating on {val_season}")
    
    # Optuna and training for each fold
    for i, (cv_train_df, cv_val_df) in enumerate(zip(cv_train_dfs, cv_val_dfs)):
        fold_start_time = datetime.now()
        logger.info(f"Optimizing fold {i+1}...")
        
        # Create objective function for this fold
        def objective(trial):
            # Define hyperparameters to tune
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'verbose': -1
            }
            
            # Get training and validation features and targets directly from DataFrames
            X_train = cv_train_df[feature_cols]
            y_train = cv_train_df['strikeouts']
            X_val = cv_val_df[feature_cols]
            y_val = cv_val_df['strikeouts']
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=10000,
                valid_sets=[val_data],
                valid_names=['validation'],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # We want to maximize the percentage of predictions within 1 strikeout
            within_1 = within_n_strikeouts(y_val, y_pred, n=1)
            
            return within_1
            
        # Optimize hyperparameters with Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        best_params = study.best_params
        best_params_list.append(best_params)
        
        logger.info(f"Best parameters for fold {i+1}: {best_params}")
        logger.info(f"Best within_1 score: {study.best_value:.4f}")
        
        # Add fixed parameters
        best_params['objective'] = 'regression'
        best_params['metric'] = 'rmse'
        best_params['verbose'] = -1
        
        # Train model with best hyperparameters on all fold training data
        X_train = cv_train_df[feature_cols]
        y_train = cv_train_df['strikeouts']
        X_val = cv_val_df[feature_cols]
        y_val = cv_val_df['strikeouts']
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        final_model = lgb.train(
            best_params,
            train_data,
            num_boost_round=10000,
            valid_sets=[val_data],
            valid_names=['validation'],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        models.append(final_model)
        
        # Evaluate on validation set
        y_pred = final_model.predict(X_val)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        within_1 = within_n_strikeouts(y_val, y_pred, n=1)
        within_2 = within_n_strikeouts(y_val, y_pred, n=2)
        
        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        metrics['within_1'].append(within_1)
        metrics['within_2'].append(within_2)
        
        # Calculate how long the fold took
        fold_time = datetime.now() - fold_start_time
        
        val_season = cv_val_df['season'].unique()[0]
        logger.info(f"Season {val_season} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        logger.info(f"Season {val_season} - Within 1 K: {within_1:.4f}, Within 2 K: {within_2:.4f}")
        logger.info(f"Fold completed in {fold_time}")
        
        # Save feature importance
        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': final_model.feature_importance(),
            'Fold': i+1,
            'Test_Season': val_season
        })
        importance_dfs.append(importance)
        
        # Create feature importance plot for this fold
        plt.figure(figsize=(12, 8))
        top_features = importance.sort_values('Importance', ascending=False).head(20)
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.title(f'Complete Model - Feature Importance - Val Season {val_season}')
        plt.tight_layout()
        
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(f'plots/complete_importance_season_{val_season}.png')
    
    # Log average CV metrics
    logger.info("\nAverage metrics across all validation seasons:")
    for metric, values in metrics.items():
        logger.info(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Determine best model based on 'within_1' metric
    best_fold_idx = np.argmax(metrics['within_1'])
    best_params = best_params_list[best_fold_idx]
    logger.info(f"\nBest model is from fold {best_fold_idx + 1} with within_1 = {metrics['within_1'][best_fold_idx]:.4f}")
    
    # Combine feature importances
    all_importance = pd.concat(importance_dfs)
    avg_importance = all_importance.groupby('Feature')['Importance'].mean().reset_index()
    avg_importance = avg_importance.sort_values('Importance', ascending=False)
    
    # Plot average feature importance
    plt.figure(figsize=(12, 10))
    plt.barh(avg_importance['Feature'].head(30), avg_importance['Importance'].head(30))
    plt.title('Average Feature Importance Across All Folds')
    plt.tight_layout()
    plt.savefig(f'plots/complete_avg_importance_{datetime.now().strftime("%Y%m%d")}.png')
    
    # Train final model on all training data
    logger.info("\nTraining final model on all training data...")
    
    X_train_all = train_df[feature_cols]
    y_train_all = train_df['strikeouts']
    
    # Create LightGBM dataset
    full_train_data = lgb.Dataset(X_train_all, label=y_train_all)
    
    # Train final model
    final_model = lgb.train(
        best_params, 
        full_train_data,
        num_boost_round=10000,
        valid_sets=None,  # No validation set for final model
        callbacks=None
    )
    
    # IMPORTANT: Final evaluation on completely separate test set
    logger.info("\nEvaluating on separate test seasons (no data leakage)...")
    X_test = test_df[feature_cols]
    y_test = test_df['strikeouts']
    
    final_preds = final_model.predict(X_test)
    final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
    final_mae = mean_absolute_error(y_test, final_preds)
    final_within_1 = within_n_strikeouts(y_test, final_preds, n=1)
    final_within_2 = within_n_strikeouts(y_test, final_preds, n=2)
    
    logger.info(f"Final model test RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")
    logger.info(f"Final model test within 1 K: {final_within_1:.4f}")
    logger.info(f"Final model test within 2 K: {final_within_2:.4f}")
    
    # Visualize predictions vs actuals
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, final_preds, alpha=0.5)
    plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
    plt.xlabel('Actual Strikeouts')
    plt.ylabel('Predicted Strikeouts')
    plt.title('Complete Model - Predicted vs Actual Strikeouts (Test Set)')
    plt.savefig(f'plots/complete_predictions_vs_actuals_{datetime.now().strftime("%Y%m%d")}.png')
    
    # Save final model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True, parents=True)
    model_filename = f'models/complete_strikeout_predictor_{datetime.now().strftime("%Y%m%d")}.txt'
    final_model.save_model(model_filename)
    
    # Save feature columns
    with open(f'models/complete_feature_columns_{datetime.now().strftime("%Y%m%d")}.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    logger.info(f"Final model saved to {model_filename}")
    
    return final_model, feature_cols

if __name__ == "__main__":
    logger.info("Starting strikeout model training with complete features...")
    model, features = train_model()
    logger.info("Complete model training complete.")