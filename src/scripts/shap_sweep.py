from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DBConfig, FileConfig, LogConfig, StrikeoutModelConfig
from src.utils import setup_logger
from src.train_model import load_dataset, split_by_year, train_lgbm

logger = setup_logger("shap_sweep", LogConfig.LOG_DIR / "shap_sweep.log")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM model and compute SHAP importances"
    )
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
    model, metrics, features = train_lgbm(train_df, test_df)

    X_train = train_df[features]
    try:
        import shap  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.error("SHAP unavailable: %s", exc)
        return

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_train)
    if isinstance(values, list):
        values = values[0]
    importance = np.abs(values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": X_train.columns, "importance": importance})
    shap_df.sort_values("importance", ascending=False, inplace=True)

    out_path = FileConfig.PLOTS_DIR / "shap_importance.csv"
    shap_df.to_csv(out_path, index=False)
    logger.info("Saved SHAP importance to %s", out_path)


if __name__ == "__main__":  # pragma: no cover
    main()
