from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DBConfig

from src.features import (
    engineer_pitcher_features,
    engineer_workload_features,
    engineer_opponent_features,
    engineer_contextual_features,
    engineer_lineup_trends,
    engineer_catcher_defense,
    engineer_batter_pitcher_history,
    build_model_features,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DBConfig.PATH,
        help="Path to SQLite DB",
    )
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

    db_path = args.db_path or DBConfig.PATH

    engineer_pitcher_features(db_path=db_path, year=args.year, rebuild=args.rebuild)
    engineer_workload_features(
        db_path=db_path,
        year=args.year,
        rebuild=args.rebuild,
    )
    engineer_opponent_features(
        db_path=db_path,
        n_jobs=args.n_jobs,
        year=args.year,
        rebuild=args.rebuild,
    )
    engineer_contextual_features(
        db_path=db_path,
        n_jobs=args.n_jobs,
        year=args.year,
        rebuild=args.rebuild,
    )
    engineer_batter_pitcher_history(
        db_path=db_path,
        year=args.year,
        rebuild=args.rebuild,
    )
    engineer_lineup_trends(
        db_path=db_path,
        n_jobs=args.n_jobs,
        year=args.year,
        rebuild=args.rebuild,
    )
    engineer_catcher_defense(
        db_path=db_path,
        n_jobs=args.n_jobs,
        year=args.year,
        rebuild=args.rebuild,
    )
    build_model_features(db_path=db_path, year=args.year, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
