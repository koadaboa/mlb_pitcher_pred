from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DBConfig, FileConfig, LogConfig
from src.train_model import (
    load_dataset,
    split_by_year,
    cross_validate_lgbm,
    train_lgbm,
    get_feature_importance,
)
from src.utils import setup_logger


logger = setup_logger(
    "train_baseline_model",
    LogConfig.LOG_DIR / "train_baseline_model.log",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train baseline LightGBM model")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DBConfig.PATH,
        help="Path to SQLite database with model_features table",
    )
    args = parser.parse_args(argv)

    df = load_dataset(args.db_path)
    if df.empty:
        logger.error("No data available for training")
        return

    train_df, test_df = split_by_year(df)

    cv_rmse = cross_validate_lgbm(train_df)

    model, metrics = train_lgbm(train_df, test_df)

    model_path = FileConfig.MODELS_DIR / "lgbm_model.txt"
    model.booster_.save_model(str(model_path))
    logger.info("Saved model to %s", model_path)

    fi_df = get_feature_importance(model)
    fi_path = FileConfig.FEATURE_IMPORTANCE_FILE
    fi_df.to_csv(fi_path, index=False)
    logger.info("Saved feature importance to %s", fi_path)

    logger.info("CV RMSE: %.4f", cv_rmse)
    logger.info("RMSE: %.4f", metrics.get("rmse", float('nan')))
    logger.info("MAE: %.4f", metrics.get("mae", float('nan')))
    logger.info("within_1_so: %.4f", metrics.get("within_1_so", float('nan')))


if __name__ == "__main__":  # pragma: no cover
    main()
