from __future__ import annotations

"""Optuna tuning script for CatBoost."""

from pathlib import Path
import json
from typing import Dict, Optional
import argparse

import numpy as np
import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

from src.config import (
    DBConfig,
    StrikeoutModelConfig,
    FileConfig,
    LogConfig,
)
from src.utils import setup_logger
from src.train_model import load_dataset, split_by_year
from pandas.api.types import is_object_dtype
from src.features.selection import select_features

logger = setup_logger("tune_catboost", LogConfig.LOG_DIR / "tune_catboost.log")


def _get_cat_features(df: pd.DataFrame, features: list[str]) -> list[int]:
    cat_cols = [
        c
        for c in features
        if is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    return [features.index(c) for c in cat_cols]


def _prepare_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> None:
    for col in cat_cols:
        df[col] = df[col].fillna("NA").astype(str)


def _objective_factory(X, y, cat_idx):
    def objective(trial: optuna.Trial) -> float:
        params = StrikeoutModelConfig.CATBOOST_BASE_PARAMS.copy()
        grid = StrikeoutModelConfig.CATBOOST_PARAM_GRID
        params.update(
            {
                "depth": trial.suggest_int("depth", *grid["depth"]),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *grid["learning_rate"]
                ),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", *grid["l2_leaf_reg"]
                ),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", *grid["bagging_temperature"]
                ),
                "random_strength": trial.suggest_float(
                    "random_strength", *grid["random_strength"]
                ),
            }
        )
        cv = TimeSeriesSplit(n_splits=StrikeoutModelConfig.OPTUNA_CV_SPLITS)
        scores = []
        for train_idx, valid_idx in cv.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            model = CatBoostRegressor(
                **params,
                iterations=StrikeoutModelConfig.FINAL_ESTIMATORS,
            )
            model.fit(
                X_train,
                y_train,
                cat_features=cat_idx,
                eval_set=(X_valid, y_valid),
                verbose=False,
            )
            preds = model.predict(X_valid)
            scores.append(np.sqrt(mean_squared_error(y_valid, preds)))
        return float(np.mean(scores))

    return objective


def tune_catboost(
    db_path: Path = DBConfig.PATH,
    *,
    n_trials: int = StrikeoutModelConfig.OPTUNA_TRIALS,
    csv_path: Optional[Path] = None,
) -> Dict[str, float]:
    df = load_dataset(db_path)
    train_df, _ = split_by_year(df)
    numeric_features, _ = select_features(train_df, StrikeoutModelConfig.TARGET_VARIABLE)
    logger.info("Using %d numeric features", len(numeric_features))
    cat_cols = [
        c
        for c in train_df.columns
        if c not in numeric_features + [StrikeoutModelConfig.TARGET_VARIABLE]
        and (
            is_object_dtype(train_df[c])
            or isinstance(train_df[c].dtype, pd.CategoricalDtype)
        )
    ]
    features = numeric_features + cat_cols
    logger.info("Using %d total features", len(features))
    _prepare_categoricals(train_df, cat_cols)
    X = train_df[features]
    y = train_df[StrikeoutModelConfig.TARGET_VARIABLE]
    cat_idx = _get_cat_features(train_df, features)

    objective = _objective_factory(X, y, cat_idx)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=StrikeoutModelConfig.OPTUNA_TIMEOUT,
    )
    logger.info("Best params: %s", study.best_params)
    out_path = FileConfig.MODELS_DIR / "catboost_best_params.json"
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info("Saved best params to %s", out_path)

    if csv_path:
        df = study.trials_dataframe()
        df.to_csv(csv_path, index=False)
        logger.info("Wrote study history to %s", csv_path)
    return study.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune CatBoost model")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DBConfig.PATH,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=StrikeoutModelConfig.OPTUNA_TRIALS,
        help="Number of Optuna trials to run",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional path to save Optuna study results as CSV",
    )
    args = parser.parse_args()
    tune_catboost(args.db_path, n_trials=args.trials, csv_path=args.csv_path)
