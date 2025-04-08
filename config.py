# config.py
class DBConfig:
    PATH = "data/pitcher_stats.db"
    BATCH_SIZE = 1000

class DataConfig:
    SEASONS = [2021, 2022, 2023, 2024, 2025]
    RATE_LIMIT_PAUSE = 5
    CHUNK_SIZE = 14

class StrikeoutModelConfig:
    RANDOM_STATE = 3
    WINDOW_SIZES = [3, 5, 10]
    DEFAULT_TRAIN_YEARS = (2021, 2022, 2023)
    DEFAULT_TEST_YEARS = (2024, 2025)
    OPTIMIZATION_METRICS = ["within_1_strikeout", "within_2_strikeouts", "over_under_accuracy"]