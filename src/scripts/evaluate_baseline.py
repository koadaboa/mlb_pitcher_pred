import argparse
import pickle
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.config import DBConfig, FileConfig, LogConfig
from src.data.utils import DBConnection, setup_logger, find_latest_file
from src.models.baseline import recent_average_predict


def within_n_strikeouts(y_true, y_pred, n=1):
    if y_true is None or y_pred is None or len(y_true) != len(y_pred):
        return np.nan
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return np.mean(np.abs(y_true_arr - np.round(y_pred_arr)) <= n)


def load_data(db_path: Path):
    with DBConnection(db_path) as conn:
        train_df = pd.read_sql_query("SELECT * FROM train_features", conn)
        test_df = pd.read_sql_query("SELECT * FROM test_features", conn)
    return train_df, test_df


def evaluate_baseline(window: int, model_prefix: str):
    logger = setup_logger("evaluate_baseline", LogConfig.LOG_DIR / "evaluate_baseline.log")
    db_path = Path(DBConfig.PATH)
    train_df, test_df = load_data(db_path)
    if train_df.empty or test_df.empty:
        logger.error("Training or test data could not be loaded from database.")
        return

    preds = recent_average_predict(train_df, test_df, window=window)
    y_true = test_df["strikeouts"].values

    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    w1 = within_n_strikeouts(y_true, preds, n=1)
    logger.info(f"Baseline RMSE: {rmse:.4f}, MAE: {mae:.4f}, within_1_so: {w1:.4f}")

    # Attempt LightGBM comparison
    model_dir = Path(FileConfig.MODELS_DIR)
    model_path = find_latest_file(model_dir, f"{model_prefix}_strikeout_model_*.txt")
    feature_path = find_latest_file(model_dir, f"{model_prefix}_feature_columns_*.pkl")
    if model_path and feature_path:
        logger.info(f"Loading LightGBM model from {model_path}")
        model = lgb.Booster(model_file=str(model_path))
        with open(feature_path, "rb") as f:
            feature_cols = pickle.load(f)
        missing = [f for f in feature_cols if f not in test_df.columns]
        if missing:
            logger.error(f"Test data missing required features for LGBM: {missing}")
            return
        lgb_preds = model.predict(test_df[feature_cols])
        lgb_rmse = np.sqrt(mean_squared_error(y_true, lgb_preds))
        lgb_mae = mean_absolute_error(y_true, lgb_preds)
        lgb_w1 = within_n_strikeouts(y_true, lgb_preds, n=1)
        logger.info(
            f"LightGBM RMSE: {lgb_rmse:.4f}, MAE: {lgb_mae:.4f}, within_1_so: {lgb_w1:.4f}"
        )
        logger.info(f"RMSE improvement vs baseline: {rmse - lgb_rmse:+.4f}")
        logger.info(f"MAE improvement vs baseline: {mae - lgb_mae:+.4f}")
        logger.info(f"Within_1_so improvement: {lgb_w1 - w1:+.4f}")
    else:
        logger.warning("LightGBM artifacts not found; skipping comparison.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recent-average baseline")
    parser.add_argument("--window", type=int, default=5, help="Number of games for average")
    parser.add_argument(
        "--model-prefix", type=str, default="test_lgb", help="Prefix for LGBM artifacts"
    )
    args = parser.parse_args()
    evaluate_baseline(window=args.window, model_prefix=args.model_prefix)

