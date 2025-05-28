from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Optional
import argparse

import numpy as np
import optuna
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    DBConfig,
    StrikeoutModelConfig,
    FileConfig,
    LogConfig,
)
from src.utils import setup_logger
from src.train_model import load_dataset, split_by_year
from src.features.selection import select_features

logger = setup_logger("tune_lightgbm", LogConfig.LOG_DIR / "tune_lightgbm.log")


def _objective_factory(X, y):
    def objective(trial: optuna.Trial) -> float:
        params = StrikeoutModelConfig.LGBM_BASE_PARAMS.copy()
        grid = StrikeoutModelConfig.LGBM_PARAM_GRID
        params.update(
            {
                "learning_rate": trial.suggest_float(
                    "learning_rate", *grid["learning_rate"]
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves", *grid["num_leaves"]
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", *grid["max_depth"]
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", *grid["min_child_samples"]
                ),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction", *grid["feature_fraction"]
                ),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", *grid["bagging_fraction"]
                ),
                "bagging_freq": trial.suggest_int(
                    "bagging_freq", *grid["bagging_freq"]
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", *grid["reg_alpha"], log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", *grid["reg_lambda"], log=True
                ),
            }
        )
        cv = TimeSeriesSplit(n_splits=StrikeoutModelConfig.OPTUNA_CV_SPLITS)
        scores = []
        for train_idx, valid_idx in cv.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            model = LGBMRegressor(
                **params,
                n_estimators=StrikeoutModelConfig.FINAL_ESTIMATORS,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[
                    lgb.early_stopping(
                        StrikeoutModelConfig.EARLY_STOPPING_ROUNDS
                    )
                ],
            )
            preds = model.predict(X_valid)
            scores.append(
                mean_squared_error(y_valid, preds, squared=False)
            )
        return float(np.mean(scores))

    return objective


def tune_lgbm(
    db_path: Path = DBConfig.PATH,
    *,
    n_trials: int = StrikeoutModelConfig.OPTUNA_TRIALS,
    csv_path: Optional[Path] = None,
) -> Dict[str, float]:
    df = load_dataset(db_path)
    train_df, _ = split_by_year(df)
    features, _ = select_features(train_df, StrikeoutModelConfig.TARGET_VARIABLE)
    X = train_df[features]
    y = train_df[StrikeoutModelConfig.TARGET_VARIABLE]

    objective = _objective_factory(X, y)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=StrikeoutModelConfig.OPTUNA_TIMEOUT,
    )
    logger.info("Best params: %s", study.best_params)
    out_path = FileConfig.MODELS_DIR / "lgbm_best_params.json"
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info("Saved best params to %s", out_path)

    if csv_path:
        df = study.trials_dataframe()
        df.to_csv(csv_path, index=False)
        logger.info("Wrote study history to %s", csv_path)
    return study.best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune LightGBM model")
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
    tune_lgbm(args.db_path, n_trials=args.trials, csv_path=args.csv_path)
