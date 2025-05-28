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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers for rolling features",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Process only games from the specified year",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop existing feature tables and recreate them",
    )
    args = parser.parse_args(argv)

    engineer_pitcher_features(
        db_path=args.db_path, year=args.year, rebuild=args.rebuild
    )
    engineer_opponent_features(
        db_path=args.db_path,
        n_jobs=args.n_jobs,
        year=args.year,
        rebuild=args.rebuild,
    )
    engineer_contextual_features(
        db_path=args.db_path,
        n_jobs=args.n_jobs,
        year=args.year,
        rebuild=args.rebuild,
    )
    build_model_features(db_path=args.db_path, year=args.year, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
