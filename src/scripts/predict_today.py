from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

import pandas as pd
import lightgbm as lgb

from src.utils import load_table_cached, setup_logger
from src.config import DBConfig, FileConfig, LogConfig

logger = setup_logger("predict_today", LogConfig.LOG_DIR / "predict_today.log")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate strikeout predictions")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DBConfig.PATH,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=FileConfig.MODELS_DIR / "lgbm_model.txt",
        help="Path to trained LightGBM model",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save predictions as CSV",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Game date to predict (YYYY-MM-DD). Defaults to today",
    )

    args = parser.parse_args(argv)

    target_date = args.date or date.today().strftime("%Y-%m-%d")

    df = load_table_cached(args.db_path, "model_features")
    df = df[df["game_date"] == target_date]
    if df.empty:
        logger.warning("No rows found for %s", target_date)
        return

    fi_path = FileConfig.FEATURE_IMPORTANCE_FILE
    if fi_path.exists():
        features = pd.read_csv(fi_path)["feature"].tolist()
    else:
        exclude = {"strikeouts", "game_pk", "game_date", "pitcher_id"}
        features = [c for c in df.columns if c not in exclude]
        logger.warning("Feature importance file missing. Using %d features", len(features))

    booster = lgb.Booster(model_file=str(args.model_file))
    preds = booster.predict(df[features])

    result = df[["game_pk", "pitcher_id", "game_date"]].copy()
    result["predicted_strikeouts"] = preds

    if args.output_csv:
        result.to_csv(args.output_csv, index=False)
    else:
        result.to_csv(sys.stdout, index=False)


if __name__ == "__main__":  # pragma: no cover
    main()
