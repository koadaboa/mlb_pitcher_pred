# config.py
from pathlib import Path
import pandas as pd
import numpy as np
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BALLPARK_COORDS = {
    'Chase Field'                   : (33.4455, -112.0667),
    'Truist Park'                   : (33.8908, -84.4671),
    'Oriole Park at Camden Yards'   : (39.2839, -76.6217),
    'Fenway Park'                   : (42.3467, -71.0972),
    'Wrigley Field'                 : (41.9484, -87.6553),
    'Guaranteed Rate Field'         : (41.8309, -87.6339),
    'Great American Ball Park'      : (39.0968, -84.5076),
    'Progressive Field'             : (41.4954, -81.6850),
    'Coors Field'                   : (39.7561, -104.9942),
    'Comerica Park'                 : (42.3390, -83.0485),
    'Minute Maid Park'              : (29.7571, -95.3557),
    'Kauffman Stadium'              : (39.0514, -94.4803),
    'Angel Stadium'                 : (33.8003, -117.8827),
    'Dodger Stadium'                : (34.0739, -118.2390),
    'loanDepot Park'                : (25.7781, -80.2197),
    'American Family Field'         : (43.0283, -87.9717),
    'Target Field'                  : (44.9817, -93.2772),
    'Citi Field'                    : (40.7571, -73.8458),
    'Yankee Stadium'                : (40.8296, -73.9262),
    'Oakland Coliseum'              : (37.7534, -122.2005),
    'Citizens Bank Park'            : (39.9057, -75.1664),
    'PNC Park'                      : (40.4469, -80.0057),
    'Petco Park'                    : (32.7076, -117.1571),
    'Oracle Park'                   : (37.7786, -122.3893),
    'T-Mobile Park'                 : (47.5914, -122.3325),
    'Busch Stadium'                 : (38.6226, -90.1928),
    'Tropicana Field'               : (27.7683, -82.6534),
    'Globe Life Field'              : (32.7510, -97.0828),
    'Rogers Centre'                 : (43.6414, -79.3894),
    'Nationals Park'                : (38.8731, -77.0074),
}

class DBConfig:
    PATH = "data/pitcher_stats.db"
    BATCH_SIZE = 5000

class DataConfig:
    SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
    RATE_LIMIT_PAUSE = 5
    CHUNK_SIZE = 1000000

class StrikeoutModelConfig:
    RANDOM_STATE = 3
    WINDOW_SIZES = [3, 5, 10]
    DEFAULT_TRAIN_YEARS = (2016, 2017, 2018, 2019, 2021, 2022, 2023)
    DEFAULT_TEST_YEARS = (2024, 2025)
    # Number of CV splits for Optuna objective
    OPTUNA_CV_SPLITS = 4
    OPTIMIZATION_METRICS = ["within_1_strikeout", "within_2_strikeouts", "over_under_accuracy"]
    TARGET_VARIABLE = 'strikeouts'
    TARGET_VARIABLE_GLOBAL_MEAN = 4.5 # Example: Add this if needed for prediction encoding fillna
    OPTUNA_TRIALS = 100
    OPTUNA_TIMEOUT = 3600
    FINAL_ESTIMATORS = 100
    EARLY_STOPPING_ROUNDS = 10
    VERBOSE_FIT = True
    TARGET_ENCODING_COLS = [
        'p_throws',         # Pitcher throwing hand (L/R)
        'opponent_team',    # Opponent team abbreviation/ID
        'home_team',        # Home team abbreviation/ID (if needed separately)
        'home_plate_umpire',           # Umpire name (if created and merged)
        'ballpark'          # Ballpark name (if created and merged)
        # Add any other categorical columns you intend to encode
    ]

    # --- Optuna / Hyperparameter Tuning Settings ---
    OPTUNA_CV_SPLITS = 4 # TimeSeriesSplit folds
    OPTUNA_TRIALS = 50 # Default number of trials (can be overridden)
    OPTUNA_TIMEOUT = 1800 # Default timeout in seconds (30 mins)
    OPTUNA_OBJECTIVE_METRIC = 'poisson' # Metric to minimize in Optuna objective

    # --- Feature Selection Settings ---
    VIF_THRESHOLD = 10.0
    SHAP_THRESHOLD = 0.001 # Example threshold for mean abs SHAP value
    SHAP_SAMPLE_FRAC = 0.1 # Fraction of data for SHAP calculation

    # --- Final Model Training Settings ---
    FINAL_ESTIMATORS = 2000 # Max estimators for final model
    EARLY_STOPPING_ROUNDS = 50 # Early stopping rounds for final model
    VERBOSE_FIT_FREQUENCY = 100 # How often to log during verbose fit

    # --- Base LightGBM Parameters (Fixed for Poisson Regression) ---
    LGBM_BASE_PARAMS = {
        'objective': 'poisson',
        'metric': 'poisson', # Use poisson deviance for evaluation
        'boosting_type': 'gbdt',
        'n_jobs': -1,       # Use all available cores
        'verbose': -1,      # Suppress verbose LightGBM logging by default
        'seed': RANDOM_STATE
        # Add other fixed params like 'device': 'gpu' if applicable
    }

    # --- LightGBM Hyperparameter Search Space (for Optuna) ---
    LGBM_PARAM_GRID = {
        'learning_rate': (0.005, 0.1),  # Log scale search
        'num_leaves': (20, 100),
        'max_depth': (3, 12),
        'min_child_samples': (5, 50),
        'feature_fraction': (0.5, 1.0), # Alias: colsample_bytree
        'bagging_fraction': (0.5, 1.0), # Alias: subsample
        'bagging_freq': (1, 7),
        'reg_alpha': (1e-8, 10.0),      # L1, Log scale search
        'reg_lambda': (1e-8, 10.0)      # L2, Log scale search
    }

class LogConfig:
    # Define Log directory relative to project root
    LOG_DIR = PROJECT_ROOT / 'logs'
    # Ensure the directory exists when config is loaded (optional but helpful)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

class FileConfig:
    # Define Model and Plot directories relative to project root
    MODELS_DIR = PROJECT_ROOT / 'models'
    PLOTS_DIR = PROJECT_ROOT / 'plots'
    DATA_DIR = PROJECT_ROOT / 'data'
    # Ensure these directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)