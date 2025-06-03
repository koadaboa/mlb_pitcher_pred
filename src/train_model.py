from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Dict
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    DBConfig,
    StrikeoutModelConfig,
    FileConfig,
    LogConfig,
)
from src.utils import DBConnection, setup_logger
from src.features.selection import select_features, filter_features_by_shap
from src.features.feature_groups import assign_feature_group

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
    holdout_year: int | None = None,
) -> Tuple[pd.DataFrame, ...]:
    """Split ``df`` into train/test sets and optional holdout set by ``game_date`` year."""
    if "game_date" not in df.columns:
        raise KeyError("game_date column missing from dataframe")

    # Filter train and test sets
    train_mask = df["game_date"].dt.year.isin(train_years)
    test_mask = df["game_date"].dt.year.isin(test_years)

    holdout_df = pd.DataFrame()
    if holdout_year is not None:
        holdout_mask = df["game_date"].dt.year == holdout_year
        holdout_df = df[holdout_mask].sort_values("game_date")
        # Exclude holdout year from train and test splits
        train_mask &= ~holdout_mask
        test_mask &= ~holdout_mask

    train_df = df[train_mask].sort_values("game_date")
    test_df = df[test_mask].sort_values("game_date")

    if holdout_year is not None:
        return train_df, test_df, holdout_df
    return train_df, test_df


def train_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = StrikeoutModelConfig.TARGET_VARIABLE,
) -> Tuple[LGBMRegressor, Dict[str, float], list[str]]:
    """Train LightGBM model and return the trained model and metrics."""
    # Select features using tree-based importance
    features, _ = select_features(
        train_df,
        target,
        prune_importance=True,
        importance_threshold=StrikeoutModelConfig.IMPORTANCE_THRESHOLD,
        importance_method="lightgbm",
    )
    features = filter_features_by_shap(features)
    logger.info("Using %d features", len(features))
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
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    within_1 = ((pd.Series(preds).round() - y_test).abs() <= 1).mean()

    metrics = {"rmse": rmse, "mae": mae, "within_1_so": within_1}
    logger.info("Evaluation metrics: %s", metrics)
    return model, metrics, features


def cross_validate_lgbm(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    target: str = StrikeoutModelConfig.TARGET_VARIABLE,
) -> float:
    """Return average RMSE using ``TimeSeriesSplit`` cross-validation."""
    if df.empty:
        raise ValueError("Training dataframe is empty")

    features, _ = select_features(
        df,
        target,
        prune_importance=True,
        importance_threshold=StrikeoutModelConfig.IMPORTANCE_THRESHOLD,
        importance_method="lightgbm",
    )
    features = filter_features_by_shap(features)
    logger.info("Using %d features", len(features))

    X = df.sort_values("game_date")[features]
    y = df.sort_values("game_date")[target]

    params = StrikeoutModelConfig.LGBM_BASE_PARAMS.copy()
    cv = TimeSeriesSplit(n_splits=n_splits)
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
                ),
                lgb.log_evaluation(StrikeoutModelConfig.VERBOSE_FIT_FREQUENCY),
            ],
        )
        preds = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        scores.append(rmse)
        logger.info("Fold RMSE: %.4f", rmse)

    avg_rmse = float(np.mean(scores))
    logger.info("Average CV RMSE: %.4f", avg_rmse)
    return avg_rmse


def get_gain_importance(model: LGBMRegressor) -> pd.DataFrame:
    """Return LightGBM gain importance with feature groups."""
    booster = model.booster_
    importance = booster.feature_importance(importance_type="gain")
    fi = pd.DataFrame({"feature": booster.feature_name(), "importance": importance})
    fi["group"] = fi["feature"].map(assign_feature_group)
    fi.sort_values("importance", ascending=False, inplace=True)
    return fi


def get_shap_importance(model: LGBMRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """Return SHAP importance averaged over absolute values."""
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("SHAP unavailable: %s", exc)
        return pd.DataFrame(columns=["feature", "importance", "group"])

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[0]
    importance = np.abs(values).mean(axis=0)
    fi = pd.DataFrame({"feature": X.columns, "importance": importance})
    fi["group"] = fi["feature"].map(assign_feature_group)
    fi.sort_values("importance", ascending=False, inplace=True)
    return fi


def main(db_path: Path | None = None) -> None:
    db_path = db_path or DBConfig.PATH
    df = load_dataset(db_path)
    if df.empty:
        logger.error("No data available for training")
        return
    train_df, test_df = split_by_year(df)
    cv_rmse = cross_validate_lgbm(
        train_df, n_splits=StrikeoutModelConfig.OPTUNA_CV_SPLITS
    )
    model, metrics, features = train_lgbm(train_df, test_df)
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
    for name, val in metrics.items():
        logger.info("%s: %.4f", name, val)


if __name__ == "__main__":
    main()
