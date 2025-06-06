from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import DBConfig, FileConfig, LogConfig, StrikeoutModelConfig
from src.train_model import (
    load_dataset,
    split_by_year,
    cross_validate_lgbm,
    train_lgbm,
    get_gain_importance,
    get_shap_importance,
)
from src.features.selection import select_features, filter_features_by_shap
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
    parser.add_argument(
        "--holdout-year",
        type=int,
        default=None,
        help="Year to hold out from training for final evaluation",
    )
    args = parser.parse_args(argv)

    df = load_dataset(args.db_path)
    if df.empty:
        logger.error("No data available for training")
        return

    split_result = split_by_year(df, holdout_year=args.holdout_year)
    if args.holdout_year is not None:
        train_df, test_df, holdout_df = split_result
    else:
        train_df, test_df = split_result
        holdout_df = pd.DataFrame()

    cv_rmse = cross_validate_lgbm(train_df)

    model, metrics, features = train_lgbm(train_df, test_df)

    holdout_metrics = {}
    if not holdout_df.empty:
        features, _ = select_features(
            train_df,
            StrikeoutModelConfig.TARGET_VARIABLE,
            prune_importance=True,
            importance_threshold=StrikeoutModelConfig.IMPORTANCE_THRESHOLD,
            importance_method="lightgbm",
        )
        features = filter_features_by_shap(features)
        X_hold = holdout_df[features]
        y_hold = holdout_df[StrikeoutModelConfig.TARGET_VARIABLE]
        preds_hold = model.predict(X_hold)
        holdout_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_hold, preds_hold))),
            "mae": float(mean_absolute_error(y_hold, preds_hold)),
            "within_1_so": float(((pd.Series(preds_hold).round() - y_hold).abs() <= 1).mean()),
        }

    model_path = FileConfig.MODELS_DIR / "lgbm_model.txt"
    model.booster_.save_model(str(model_path))
    logger.info("Saved model to %s", model_path)

    fi_df = get_gain_importance(model)
    fi_path = FileConfig.FEATURE_IMPORTANCE_FILE
    fi_df.to_csv(fi_path, index=False)
    logger.info("Saved feature importance to %s", fi_path)

    shap_df = get_shap_importance(model, train_df[features])
    shap_path = FileConfig.SHAP_IMPORTANCE_FILE
    shap_df.to_csv(shap_path, index=False)
    logger.info("Saved SHAP importance to %s", shap_path)

    logger.info("CV RMSE: %.4f", cv_rmse)
    logger.info("RMSE: %.4f", metrics.get("rmse", float('nan')))
    logger.info("MAE: %.4f", metrics.get("mae", float('nan')))
    logger.info("within_1_so: %.4f", metrics.get("within_1_so", float('nan')))
    if holdout_metrics:
        logger.info(
            "Holdout %d RMSE: %.4f", args.holdout_year, holdout_metrics.get("rmse", float("nan"))
        )
        logger.info(
            "Holdout %d MAE: %.4f", args.holdout_year, holdout_metrics.get("mae", float("nan"))
        )
        logger.info(
            "Holdout %d within_1_so: %.4f",
            args.holdout_year,
            holdout_metrics.get("within_1_so", float("nan")),
        )


if __name__ == "__main__":  # pragma: no cover
    main()
