from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    DBConfig,
    StrikeoutModelConfig,
    FileConfig,
    LogConfig,
)
from src.utils import setup_logger
from pandas.api.types import is_object_dtype
from src.train_model import load_dataset, split_by_year
from src.features.selection import select_features

logger = setup_logger("train_catboost", LogConfig.LOG_DIR / "train_catboost.log")


def _get_cat_features(df: pd.DataFrame, features: List[str]) -> List[int]:
    """Return indices of categorical columns in ``features``."""
    cat_cols = [
        c
        for c in features
        if is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    return [features.index(c) for c in cat_cols]


def _prepare_categoricals(df: pd.DataFrame, cat_cols: List[str]) -> None:
    """Fill missing categorical values and ensure string dtype."""
    for col in cat_cols:
        df[col] = df[col].fillna("NA").astype(str)


def train_catboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = StrikeoutModelConfig.TARGET_VARIABLE,
) -> Tuple[CatBoostRegressor, Dict[str, float]]:
    numeric_features, _ = select_features(train_df, target)
    logger.info("Using %d numeric features", len(numeric_features))
    cat_cols = [
        c
        for c in train_df.columns
        if c not in numeric_features + [target]
        and (
            is_object_dtype(train_df[c])
            or isinstance(train_df[c].dtype, pd.CategoricalDtype)
        )
    ]
    features = numeric_features + cat_cols
    logger.info("Using %d total features", len(features))

    _prepare_categoricals(train_df, cat_cols)
    _prepare_categoricals(test_df, cat_cols)

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    cat_indices = _get_cat_features(train_df, features)

    params = {
        "loss_function": "RMSE",
        "random_seed": StrikeoutModelConfig.RANDOM_STATE,
        "verbose": False,
    }
    model = CatBoostRegressor(
        **params,
        iterations=StrikeoutModelConfig.FINAL_ESTIMATORS,
    )

    model.fit(X_train, y_train, cat_features=cat_indices, eval_set=(X_test, y_test), verbose=False)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
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
    model, metrics = train_catboost(train_df, test_df)
    model_path = FileConfig.MODELS_DIR / "catboost_model.cbm"
    model.save_model(str(model_path))
    logger.info("Saved model to %s", model_path)
    for name, val in metrics.items():
        logger.info("%s: %.4f", name, val)


if __name__ == "__main__":
    main()
