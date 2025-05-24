"""Run all feature engineering steps."""
from src.features import (
    engineer_pitcher_features,
    engineer_opponent_features,
    engineer_contextual_features,
)
from src.utils import setup_logger
from src.config import LogConfig

logger = setup_logger(
    "run_feature_engineering",
    LogConfig.LOG_DIR / "run_feature_engineering.log",
)


def main() -> None:
    logger.info("Starting feature engineering pipeline")
    engineer_pitcher_features()
    engineer_opponent_features()
    engineer_contextual_features()
    logger.info("Feature engineering pipeline complete")


if __name__ == "__main__":
    main()
