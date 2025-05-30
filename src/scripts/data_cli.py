"""Command line entry point for data fetching utilities."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from src.utils import ensure_dir, setup_logger
from src.config import LogConfig
from .data_fetcher import DataFetcher

logger = setup_logger("data_cli", log_file=Path(LogConfig.LOG_DIR) / "data_cli.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MLB Statcast data OR probable pitchers via API.")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD) for API scrape OR single-date historical fetch.")
    parser.add_argument("--seasons", type=int, nargs="+", default=None, help="Seasons for historical backfill (default: from config or 2019-today).")
    parser.add_argument("--parallel", action="store_true", help="Use parallel fetching for historical pitcher AND batter data.")
    parser.add_argument("--mlb-api", action="store_true", help="ONLY fetch probable pitchers via API for the SINGLE date specified by --date.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.info("DEBUG logging enabled.")
    else:
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)

    if args.mlb_api and not args.date:
        logger.error("--mlb-api requires --date.")
        return 1
    if args.mlb_api and args.seasons:
        logger.warning("--seasons argument is ignored when --mlb-api is used.")
    if args.date and args.seasons and not args.mlb_api:
        logger.warning("--date argument is ignored when --seasons is used (unless in single-date mode without --mlb-api). Effective end date is calculated.")

    ensure_dir(Path(LogConfig.LOG_DIR))
    ensure_dir(Path("data"))
    ensure_dir(Path("data/.checkpoints"))

    logger.info("--- Initializing MLB Data Fetcher ---")
    fetcher = DataFetcher(args)
    success = fetcher.run()
    if success:
        logger.info("--- Data Fetching Script Finished Successfully ---")
        return 0
    logger.error("--- Data Fetching Script Finished With Errors ---")
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
