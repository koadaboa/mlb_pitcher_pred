from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    DBConfig,
    StrikeoutModelConfig,
    FileConfig,
    LogConfig,
)
from src.utils import setup_logger
from src.train_model import load_dataset, split_by_year
from src.features.selection import select_features

logger = setup_logger("train_xgb_model", LogConfig.LOG_DIR / "train_xgb_model.log")


def train_xgb(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = StrikeoutModelConfig.TARGET_VARIABLE,
) -> Tuple[XGBRegressor, Dict[str, float]]:
    features, _ = select_features(train_df, target)
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    params = StrikeoutModelConfig.XGB_BASE_PARAMS.copy()
    model = XGBRegressor(
        **params,
        n_estimators=StrikeoutModelConfig.FINAL_ESTIMATORS,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    within_1 = ((pd.Series(preds).round() - y_test).abs() <= 1).mean()

    metrics = {"rmse": rmse, "mae": mae, "within_1_so": within_1}
    logger.info("Evaluation metrics: %s", metrics)
    return model, metrics


def main(db_path: Path | None = None) -> None:
    db_path = db_path or DBConfig.PATH
    df = load_dataset(db_path)
    if df.empty:
        logger.error("No data available for training")
        return
    train_df, test_df = split_by_year(df)
    model, metrics = train_xgb(train_df, test_df)
    model_path = FileConfig.MODELS_DIR / "xgb_model.json"
    model.save_model(str(model_path))
    logger.info("Saved model to %s", model_path)
    for name, val in metrics.items():
        logger.info("%s: %.4f", name, val)


if __name__ == "__main__":
    main()
