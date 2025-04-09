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

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.utils import setup_logger, DBConnection

# Setup logger
logger = setup_logger('train_complete_model')

def within_n_strikeouts(y_true, y_pred, n=1):
    """Calculate percentage of predictions within n strikeouts of actual value"""
    within_n = np.abs(y_true - np.round(y_pred)) <= n
    return np.mean(within_n)

def load_data():
    """Load combined predictive features"""
    logger.info("Loading combined feature data...")
    with DBConnection() as conn:
        query = "SELECT * FROM combined_predictive_features"
        df = pd.read_sql_query(query, conn)
    
    logger.info(f"Loaded {len(df)} rows of combined features")
    return df

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
        'player_name_opp', 'opposing_team', 'game_month'
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
    """Train LightGBM model with all features"""
    # Load and prepare data
    df = load_data()
    df, feature_cols = prepare_data(df)
    
    # Get time series splits by season
    splits = time_series_cv_by_season(df)
    
    # Metrics storage
    metrics = {
        'rmse': [], 'mae': [], 
        'within_1': [], 'within_2': []
    }
    
    # Store best parameters and models for each fold
    best_params_list = []
    models = []
    
    # Save feature importance for each fold
    importance_dfs = []
    
    # Optuna and training for each fold
    for i, (train_idx, test_idx) in enumerate(splits):
        fold_start_time = datetime.now()
        logger.info(f"Optimizing fold {i+1}...")
        
        train_seasons = df.iloc[train_idx]['season'].unique()
        test_season = df.iloc[test_idx]['season'].unique()[0]
        logger.info(f"Training on seasons {train_seasons}, testing on {test_season}")
        logger.info(f"Training data: {len(train_idx)} rows, Test data: {len(test_idx)} rows")
        
        # Create a validation set from training data
        validation_idx = []
        for pitcher_id in df.iloc[train_idx]['pitcher_id'].unique():
            pitcher_data = df[(df['pitcher_id'] == pitcher_id) & 
                              (df.index.isin(train_idx))].sort_values('game_date')
            val_size = max(1, int(len(pitcher_data) * 0.2))
            validation_idx.extend(pitcher_data.index[-val_size:])
        
        # Create train/val split ensuring time order is preserved
        actual_train_idx = [idx for idx in train_idx if idx not in validation_idx]
        
        logger.info(f"Final training data: {len(actual_train_idx)} rows, Validation data: {len(validation_idx)} rows")
        
        # Optimize hyperparameters with Optuna
        study = optuna.create_study(direction='maximize')
        objective_func = lambda trial: objective(trial, df, feature_cols, actual_train_idx, validation_idx)
        
        # Limit to 50 trials
        study.optimize(objective_func, n_trials=50)
        
        best_params = study.best_params
        best_params_list.append(best_params)
        
        logger.info(f"Best parameters for fold {i+1}: {best_params}")
        logger.info(f"Best within_1 score: {study.best_value:.4f}")
        
        # Add fixed parameters
        best_params['objective'] = 'regression'
        best_params['metric'] = 'rmse'
        best_params['verbose'] = -1
        
        # Train model with best hyperparameters on all training data
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx]['strikeouts']
        X_test = df.iloc[test_idx][feature_cols]
        y_test = df.iloc[test_idx]['strikeouts']
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        final_model = lgb.train(
            best_params,
            train_data,
            num_boost_round=10000,
            valid_sets=[test_data],
            valid_names=['test'],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        models.append(final_model)
        
        # Make predictions
        y_pred = final_model.predict(X_test)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        within_1 = within_n_strikeouts(y_test, y_pred, n=1)
        within_2 = within_n_strikeouts(y_test, y_pred, n=2)
        
        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        metrics['within_1'].append(within_1)
        metrics['within_2'].append(within_2)
        
        # Calculate how long the fold took
        fold_time = datetime.now() - fold_start_time
        
        logger.info(f"Season {test_season} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        logger.info(f"Season {test_season} - Within 1 K: {within_1:.4f}, Within 2 K: {within_2:.4f}")
        logger.info(f"Fold completed in {fold_time}")
        
        # Save feature importance
        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': final_model.feature_importance(),
            'Fold': i+1,
            'Test_Season': test_season
        })
        importance_dfs.append(importance)
        
        # Create feature importance plot for this fold
        plt.figure(figsize=(12, 8))
        top_features = importance.sort_values('Importance', ascending=False).head(20)
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.title(f'Complete Model - Feature Importance - Test Season {test_season}')
        plt.tight_layout()
        
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(f'plots/complete_importance_season_{test_season}.png')
    
    # Log average metrics
    logger.info("\nAverage metrics across all test seasons:")
    for metric, values in metrics.items():
        logger.info(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Determine best fold model based on 'within_1' metric
    best_fold_idx = np.argmax(metrics['within_1'])
    best_params = best_params_list[best_fold_idx]
    logger.info(f"\nBest model is from fold {best_fold_idx + 1} with within_1 = {metrics['within_1'][best_fold_idx]:.4f}")
    
    # Combine all feature importances and analyze
    all_importance = pd.concat(importance_dfs)
    avg_importance = all_importance.groupby('Feature')['Importance'].mean().reset_index()
    avg_importance = avg_importance.sort_values('Importance', ascending=False)
    
    # Plot average feature importance
    plt.figure(figsize=(12, 10))
    plt.barh(avg_importance['Feature'].head(30), avg_importance['Importance'].head(30))
    plt.title('Average Feature Importance Across All Folds')
    plt.tight_layout()
    plt.savefig(f'plots/complete_avg_importance_{datetime.now().strftime("%Y%m%d")}.png')
    
    # Train final model on all data using best hyperparameters
    logger.info("\nTraining final model on all data using best hyperparameters...")
    
    # Create a validation set from the most recent 10% of each pitcher's data
    all_validation_idx = []
    for pitcher_id in df['pitcher_id'].unique():
        pitcher_data = df[df['pitcher_id'] == pitcher_id].sort_values('game_date')
        val_size = max(1, int(len(pitcher_data) * 0.1))
        all_validation_idx.extend(pitcher_data.index[-val_size:])
    
    # Create train/val split for final model
    final_train_idx = [idx for idx in df.index if idx not in all_validation_idx]
    X_train_all = df.iloc[final_train_idx][feature_cols]
    y_train_all = df.iloc[final_train_idx]['strikeouts']
    X_val_all = df.iloc[all_validation_idx][feature_cols]
    y_val_all = df.iloc[all_validation_idx]['strikeouts']
    
    # Create LightGBM datasets
    full_train_data = lgb.Dataset(X_train_all, label=y_train_all)
    full_val_data = lgb.Dataset(X_val_all, label=y_val_all, reference=full_train_data)
    
    # Train final model
    final_model = lgb.train(
        best_params, 
        full_train_data,
        num_boost_round=10000,
        valid_sets=[full_val_data],
        valid_names=['validation'],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Evaluate on validation set
    final_preds = final_model.predict(X_val_all)
    final_within_1 = within_n_strikeouts(y_val_all, final_preds, n=1)
    final_within_2 = within_n_strikeouts(y_val_all, final_preds, n=2)
    final_rmse = np.sqrt(mean_squared_error(y_val_all, final_preds))
    final_mae = mean_absolute_error(y_val_all, final_preds)
    
    logger.info(f"Final model validation RMSE: {final_rmse:.4f}, MAE: {final_mae:.4f}")
    logger.info(f"Final model validation within 1 K: {final_within_1:.4f}")
    logger.info(f"Final model validation within 2 K: {final_within_2:.4f}")
    
    # Visualize predictions vs actuals
    plt.figure(figsize=(10, 8))
    plt.scatter(y_val_all, final_preds, alpha=0.5)
    plt.plot([0, max(y_val_all)], [0, max(y_val_all)], 'r--')
    plt.xlabel('Actual Strikeouts')
    plt.ylabel('Predicted Strikeouts')
    plt.title('Complete Model - Predicted vs Actual Strikeouts')
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