# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    OPTIMIZATION_METRICS = ["within_1_strikeout", "within_2_strikeouts", "over_under_accuracy"]
    TARGET_VARIABLE = 'strikeouts'
    TARGET_VARIABLE_GLOBAL_MEAN = 4.5 # Example: Add this if needed for prediction encoding fillna
    OPTUNA_TRIALS = 100
    OPTUNA_TIMEOUT = 3600
    FINAL_ESTIMATORS = 100
    EARLY_STOPPING_ROUNDS = 10
    VERBOSE_FIT = True
    # --- ADD THIS LINE ---
    TARGET_ENCODING_COLS = [
        'p_throws',         # Pitcher throwing hand (L/R)
        'opponent_team',    # Opponent team abbreviation/ID
        'home_team',        # Home team abbreviation/ID (if needed separately)
        'home_plate_umpire',           # Umpire name (if created and merged)
        'ballpark'          # Ballpark name (if created and merged)
        # Add any other categorical columns you intend to encode
    ]
    # --- END ADDITION ---

class LogConfig:
    # Define Log directory relative to project root
    LOG_DIR = PROJECT_ROOT / 'logs'
    # Ensure the directory exists when config is loaded (optional but helpful)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

class FileConfig:
    # Define Model and Plot directories relative to project root
    MODELS_DIR = PROJECT_ROOT / 'models'
    PLOTS_DIR = PROJECT_ROOT / 'plots'
    # Ensure these directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)