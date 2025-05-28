from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Dict

import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    DBConfig,
    StrikeoutModelConfig,
    FileConfig,
    LogConfig,
)
from src.utils import DBConnection, setup_logger
from src.features.selection import select_features

logger = setup_logger("train_model", LogConfig.LOG_DIR / "train_model.log")


def load_dataset(db_path: Path = DBConfig.PATH) -> pd.DataFrame:
    """Return the full model features table as a DataFrame."""
    with DBConnection(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM model_features", conn)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def split_by_year(
    df: pd.DataFrame,
    train_years: Sequence[int] = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS,
    test_years: Sequence[int] = StrikeoutModelConfig.DEFAULT_TEST_YEARS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train and test sets based on the ``game_date`` year."""
    if "game_date" not in df.columns:
        raise KeyError("game_date column missing from dataframe")
    train_df = df[df["game_date"].dt.year.isin(train_years)].sort_values("game_date")
    test_df = df[df["game_date"].dt.year.isin(test_years)].sort_values("game_date")
    return train_df, test_df


def train_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = StrikeoutModelConfig.TARGET_VARIABLE,
) -> Tuple[LGBMRegressor, Dict[str, float]]:
    """Train LightGBM model and return the trained model and metrics."""
    features, _ = select_features(train_df, target)
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]


    params = StrikeoutModelConfig.LGBM_BASE_PARAMS.copy()
    model = LGBMRegressor(
        **params,
        n_estimators=StrikeoutModelConfig.FINAL_ESTIMATORS,
    )

    # Use LightGBM callbacks for early stopping and periodic logging
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(StrikeoutModelConfig.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(StrikeoutModelConfig.VERBOSE_FIT_FREQUENCY),
        ],
    )

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    within_1 = ((pd.Series(preds).round() - y_test).abs() <= 1).mean()

    metrics = {"rmse": rmse, "mae": mae, "within_1_so": within_1}
    logger.info("Evaluation metrics: %s", metrics)
    return model, metrics


def get_feature_importance(model: LGBMRegressor) -> pd.DataFrame:
    """Return feature importance sorted by gain."""
    booster = model.booster_
    importance = booster.feature_importance(importance_type="gain")
    fi = pd.DataFrame({"feature": booster.feature_name(), "importance": importance})
    fi.sort_values("importance", ascending=False, inplace=True)
    return fi


def main(db_path: Path | None = None) -> None:
    db_path = db_path or DBConfig.PATH
    df = load_dataset(db_path)
    if df.empty:
        logger.error("No data available for training")
        return
    train_df, test_df = split_by_year(df)
    model, metrics = train_lgbm(train_df, test_df)
    model_path = FileConfig.MODELS_DIR / "lgbm_model.txt"
    model.booster_.save_model(str(model_path))
    logger.info("Saved model to %s", model_path)
    fi_df = get_feature_importance(model)
    fi_path = FileConfig.FEATURE_IMPORTANCE_FILE
    fi_df.to_csv(fi_path, index=False)
    logger.info("Saved feature importance to %s", fi_path)
    for name, val in metrics.items():
        logger.info("%s: %.4f", name, val)


if __name__ == "__main__":
    main()
