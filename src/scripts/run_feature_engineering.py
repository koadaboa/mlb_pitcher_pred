from __future__ import annotations

import argparse
from pathlib import Path

from src.features import (
    engineer_pitcher_features,
    engineer_opponent_features,
    engineer_contextual_features,
    build_model_features,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument("--db-path", type=Path, default=None, help="Path to SQLite DB")
    args = parser.parse_args(argv)

    engineer_pitcher_features(db_path=args.db_path)
    engineer_opponent_features(db_path=args.db_path)
    engineer_contextual_features(db_path=args.db_path)
    build_model_features(db_path=args.db_path)


if __name__ == "__main__":
    main()
