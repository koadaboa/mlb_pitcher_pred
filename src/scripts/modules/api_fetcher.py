"""Helpers for MLB API calls for probable pitchers."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import logging
import pandas as pd

from src.data.mlb_api import scrape_probable_pitchers
from .store_utils import store_data_to_sql
from .checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


def fetch_probable_pitchers(
    target_date: date,
    db_path: Path,
    checkpoint_manager: CheckpointManager,
    failed_api_fetches: set,
) -> bool:
    target_date_str = target_date.strftime("%Y-%m-%d")
    logger.info("Starting probable pitcher fetch via API for: %s", target_date_str)
    try:
        daily_pitcher_data = scrape_probable_pitchers(target_date_str)
        if daily_pitcher_data is None:
            logger.warning("scrape_probable_pitchers returned None for %s. Treating as no data.", target_date_str)
            daily_pitcher_data = []
        elif not isinstance(daily_pitcher_data, list):
            logger.error("scrape_probable_pitchers returned unexpected type: %s for %s.", type(daily_pitcher_data), target_date_str)
            failed_api_fetches.add(target_date_str)
            return False
        if daily_pitcher_data:
            pdf = pd.DataFrame(daily_pitcher_data)
            expected_cols = [
                "game_date",
                "game_pk",
                "home_team_abbr",
                "away_team_abbr",
                "home_probable_pitcher_name",
                "home_probable_pitcher_id",
                "away_probable_pitcher_name",
                "away_probable_pitcher_id",
            ]
            if not all(col in pdf.columns for col in expected_cols):
                logger.error("API response for %s missing expected columns. Found: %s", target_date_str, pdf.columns.tolist())
                failed_api_fetches.add(target_date_str)
                return False
            logger.info("Storing %d probable pitcher entries for %s (replacing existing table)...", len(pdf), target_date_str)
            success = store_data_to_sql(pdf, "mlb_api", db_path, if_exists="replace")
            if success:
                checkpoint_manager.add_processed_mlb_api_date(target_date_str)
                checkpoint_manager.save_overall_checkpoint()
                logger.info("Successfully stored probable pitcher data for %s.", target_date_str)
                return True
            logger.error("Failed to store probable pitcher data for %s.", target_date_str)
            failed_api_fetches.add(target_date_str)
            return False
        logger.info("No probable pitchers found via API for %s.", target_date_str)
        checkpoint_manager.add_processed_mlb_api_date(target_date_str)
        checkpoint_manager.save_overall_checkpoint()
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error during probable pitcher fetch/store for %s: %s", target_date_str, exc, exc_info=True)
        failed_api_fetches.add(target_date_str)
        return False
