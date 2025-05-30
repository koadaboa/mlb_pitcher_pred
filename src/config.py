# config.py
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Database Schema Constants ---
# Table Names
MLB_BOXSCORES_TABLE = 'mlb_boxscores'
STATCAST_PITCHERS_TABLE = 'statcast_pitchers'
STATCAST_BATTERS_TABLE = 'statcast_batters'
STATCAST_STARTING_PITCHERS_TABLE = 'statcast_starting_pitchers'


# Team Mapping Columns 
TEAM_NAME_MAP_FULL_NAME_COL = 'team_name'
TEAM_NAME_MAP_ABBR_COL = 'team_abbr'

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

# Approximate run-scoring park factors indexed by stadium name. Values above 100
# indicate a hitter-friendly environment while numbers below 100 favor
# pitchers.  These are coarse averages and can be refined when more granular
# data is available.
BALLPARK_FACTORS = {
    'Chase Field'                   : 99,
    'Truist Park'                   : 101,
    'Oriole Park at Camden Yards'   : 97,
    'Fenway Park'                   : 104,
    'Wrigley Field'                 : 100,
    'Guaranteed Rate Field'         : 102,
    'Great American Ball Park'      : 103,
    'Progressive Field'             : 98,
    'Coors Field'                   : 115,
    'Comerica Park'                 : 99,
    'Minute Maid Park'              : 101,
    'Kauffman Stadium'              : 98,
    'Angel Stadium'                 : 99,
    'Dodger Stadium'                : 100,
    'loanDepot Park'                : 96,
    'American Family Field'         : 101,
    'Target Field'                  : 99,
    'Citi Field'                    : 98,
    'Yankee Stadium'                : 103,
    'Oakland Coliseum'              : 95,
    'Citizens Bank Park'            : 103,
    'PNC Park'                      : 97,
    'Petco Park'                    : 96,
    'Oracle Park'                   : 95,
    'T-Mobile Park'                 : 97,
    'Busch Stadium'                 : 98,
    'Tropicana Field'               : 97,
    'Globe Life Field'              : 100,
    'Rogers Centre'                 : 102,
    'Nationals Park'                : 100,
}

class DBConfig:
    # Use an absolute path so scripts work from any CWD
    PATH = PROJECT_ROOT / "data" / "pitcher_stats.db"
    BATCH_SIZE = 5000

class DataConfig:
    SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
    RATE_LIMIT_PAUSE = 5
    CHUNK_SIZE = 1000000
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', os.cpu_count() or 1))

class StrikeoutModelConfig:
    RANDOM_STATE = 3
    # Dramatically smaller windows to keep feature counts manageable
    # Expanded windows to provide more temporal context
    WINDOW_SIZES = [3, 5, 10, 20, 50, 100]
    # Halflife used for exponentially weighted moving averages
    EWM_HALFLIFE = 7
    # Limit which numeric columns get rolling stats to avoid huge tables
    PITCHER_ROLLING_COLS = [
        "strikeouts",
        "pitches",
        "fip",
        "csw_pct",
        "swinging_strike_rate",
        "first_pitch_strike_rate",
        "fastball_pct",
        "fastball_whiff_rate",
        "slider_pct",
        "curve_pct",
        "changeup_pct",
        "cutter_pct",
        "sinker_pct",
        "splitter_pct",
        "slider_whiff_rate",
        "curve_whiff_rate",
        "changeup_whiff_rate",
        "cutter_whiff_rate",
        "sinker_whiff_rate",
        "splitter_whiff_rate",
        "avg_release_speed",
        "max_release_speed",
        "avg_spin_rate",
        "offspeed_to_fastball_ratio",
        "fastball_then_breaking_rate",
        "unique_pitch_types",
    ]
    CONTEXT_ROLLING_COLS = [
        "strikeouts",
        "pitches",
        "temp",
        "wind_speed",
        "elevation",
        "humidity",
        "park_factor",
        "team_k_rate",
        "bat_whiff_rate",
        "bat_avg",
        "bat_obp",
        "bat_slugging",
        "bat_ops",
        "bat_woba",
        "team_ops_vs_LHP",
        "team_ops_vs_RHP",
        "bat_ops_vs_LHP",
        "bat_ops_vs_RHP",
        "bat_k_rate_vs_LHP",
        "bat_k_rate_vs_RHP",
    ]
    # Numeric columns that may be used without rolling (known before the game)
    ALLOWED_BASE_NUMERIC_COLS = [
        "temp",
        "wind_speed",
        "elevation",
        "rest_days",
        "humidity",
        "park_factor",
        "team_k_rate",
        "day_of_week",
        "travel_distance",
    ]
    DEFAULT_TRAIN_YEARS = (2016, 2017, 2018, 2019, 2021, 2022, 2023)
    DEFAULT_TEST_YEARS = (2024, 2025)
    TARGET_VARIABLE = "strikeouts"

    # --- LightGBM Hyperparameter Defaults ---
    LGBM_BASE_PARAMS = {
        'objective': 'poisson',
        'learning_rate': 0.03786686371695319,
        'num_leaves': 82,
        'max_depth': 3,
        'min_child_samples': 40,
        'feature_fraction': 0.6561758743969195,
        'bagging_fraction': 0.6103902823033777,
        'bagging_freq': 2,
        'reg_alpha': 0.33907972084342536,
        'reg_lambda': 8.296994158031719,
        "seed": RANDOM_STATE,
    }

    # Hyperparameter ranges widened to explore a larger search space
    LGBM_PARAM_GRID = {
        "learning_rate": (0.005, 0.3),
        "num_leaves": (16, 128),
        "max_depth": (3, 12),
        "min_child_samples": (1, 100),
        "feature_fraction": (0.4, 1.0),
        "bagging_fraction": (0.4, 1.0),
        "bagging_freq": (0, 10),
        "reg_alpha": (1e-3, 20.0),
        "reg_lambda": (1e-3, 20.0),
    }

    # --- XGBoost Defaults and Search Space ---
    XGB_BASE_PARAMS = {
        "objective": "count:poisson",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
    }

    XGB_PARAM_GRID = {
        "learning_rate": (0.005, 0.3),
        "max_depth": (3, 12),
        "min_child_weight": (1, 20),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (1e-3, 20.0),
        "reg_lambda": (1e-3, 20.0),
    }

    # --- CatBoost Defaults and Search Space ---
    CATBOOST_BASE_PARAMS = {
        "loss_function": "RMSE",
        "random_seed": RANDOM_STATE,
        "verbose": False,
        'depth': 6, 
        'learning_rate': 0.06653022135887977, 
        'l2_leaf_reg': 4.1143640300797255, 
        'bagging_temperature': 0.4708062119340739, 
        'random_strength': 7.216807176944511
    }

    CATBOOST_PARAM_GRID = {
        "depth": (4, 10),
        "learning_rate": (0.01, 0.3),
        "l2_leaf_reg": (1.0, 10.0),
        "bagging_temperature": (0.0, 1.0),
        "random_strength": (0.0, 10.0),
    }

    OPTUNA_TRIALS = 50
    OPTUNA_TIMEOUT = 1800  # seconds
    OPTUNA_CV_SPLITS = 5

    FINAL_ESTIMATORS = 500
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_FIT_FREQUENCY = 20

    IMPORTANCE_THRESHOLD = 0.02

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
    FEATURE_IMPORTANCE_FILE = MODELS_DIR / 'feature_importance.csv'
    # Ensure these directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
