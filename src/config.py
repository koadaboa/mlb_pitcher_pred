# config.py
from pathlib import Path
import pandas as pd
import numpy as np
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Database Schema Constants ---
# Table Names
MLB_BOXSCORES_TABLE = 'mlb_boxscores'
STATCAST_PITCHERS_TABLE = 'statcast_pitchers'
STATCAST_BATTERS_TABLE = 'statcast_batters'


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

class DBConfig:
    PATH = "data/pitcher_stats.db"
    BATCH_SIZE = 5000

class DataConfig:
    SEASONS = [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
    RATE_LIMIT_PAUSE = 5
    CHUNK_SIZE = 1000000

class StrikeoutModelConfig:
    RANDOM_STATE = 3
    WINDOW_SIZES = [3, 5, 10, 25]
    DEFAULT_TRAIN_YEARS = (2016, 2017, 2018, 2019, 2021, 2022, 2023)
    DEFAULT_TEST_YEARS = (2024, 2025)

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