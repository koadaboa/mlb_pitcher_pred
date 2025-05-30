"""Helpers for fetching pitcher statcast data."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List

import pandas as pd
import pybaseball as pb

from src.utils import DBConnection
from .fetch_utils import dedup_pitch_df, filter_regular_season

import logging

logger = logging.getLogger(__name__)


def fetch_pitcher_single_date(
    pitcher_id: int,
    name: str,
    target_date_obj: date,
    db_path: Path,
    fetch_with_retries,
    problematic_ids: set,
) -> pd.DataFrame:
    """Fetch statcast data for a single pitcher on a single date."""
    target_date_str = target_date_obj.strftime("%Y-%m-%d")
    target_season = target_date_obj.year
    data_exists = False
    try:
        with DBConnection(db_path) as conn:
            query = "SELECT COUNT(*) FROM statcast_pitchers WHERE DATE(game_date) = ? AND pitcher = ?"
            cursor = conn.cursor()
            pid_primitive = pitcher_id.item() if hasattr(pitcher_id, "item") else pitcher_id
            cursor.execute(query, (target_date_str, pid_primitive))
            count = cursor.fetchone()[0]
            if count > 0:
                data_exists = True
    except Exception as exc:
        logger.warning(
            "DB check failed for pitcher %s (%s) on %s: %s", name, pitcher_id, target_date_str, exc
        )
    if data_exists:
        logger.debug(" -> Skipping P fetch %s (%s): Data already exists for %s", name, pitcher_id, target_date_str)
        return pd.DataFrame()
    logger.debug(" -> Fetch P %s (%s) for single date: %s", name, pitcher_id, target_date_str)
    pd_data = fetch_with_retries(pb.statcast_pitcher, target_date_str, target_date_str, pitcher_id)
    pd_data = filter_regular_season(pd_data)
    if pd_data is None:
        logger.error(" -> Error fetching P %s (%s) single date %s after retries.", name, pitcher_id, target_date_str)
        problematic_ids.add(pitcher_id)
        return pd.DataFrame()
    if not pd_data.empty:
        try:
            pid_primitive = pitcher_id.item() if hasattr(pitcher_id, "item") else pitcher_id
            pd_data["pitcher_id"] = pid_primitive
            pd_data["season"] = target_season
            numeric_cols = ["release_speed", "release_spin_rate", "launch_speed", "launch_angle"]
            for col in numeric_cols:
                if col in pd_data.columns:
                    pd_data[col] = pd.to_numeric(pd_data[col], errors="coerce")
            essential_cols = ["game_pk", "pitcher", "batter", "pitch_number"]
            if all(col in pd_data.columns for col in essential_cols):
                pd_data = pd_data.dropna(subset=essential_cols)
                pd_data = dedup_pitch_df(pd_data)
            else:
                logger.warning(
                    "Missing essential columns in fetched data for P %s (%s) on %s", name, pitcher_id, target_date_str
                )
            logger.debug(" -> Fetched %d rows for P %s single date", len(pd_data), name)
            return pd_data
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Error processing fetched data for P %s single date %s: %s", name, target_date_str, exc)
            problematic_ids.add(pitcher_id)
            return pd.DataFrame()
    logger.debug(" -> No data found for P %s (%s) on %s", name, pitcher_id, target_date_str)
    return pd.DataFrame()


def fetch_pitcher_historical(
    pitcher_id: int,
    name: str,
    seasons_list: List[int],
    end_date_limit: date,
    db_path: Path,
    fetch_with_retries,
    checkpoint_manager,
    problematic_ids: set,
) -> pd.DataFrame:
    """Fetch historical statcast data for a pitcher across seasons."""
    if checkpoint_manager.is_pitcher_processed(pitcher_id):
        logger.debug(" -> Skipping P fetch %s (%s): Already processed per checkpoint.", name, pitcher_id)
        return pd.DataFrame()

    all_data = []
    logger.debug(" -> Hist fetch P: %s (%s) for seasons %s", name, pitcher_id, seasons_list)
    relevant_seasons = [s for s in seasons_list if s <= end_date_limit.year]
    fetch_failed = False
    for season in relevant_seasons:
        start_dt = date(season, 3, 1)
        end_dt = end_date_limit if season == end_date_limit.year else date(season, 11, 30)
        if start_dt > end_dt:
            logger.debug(
                " -> Skipping season %s for P %s: Start date %s is after end date %s.",
                season,
                name,
                start_dt,
                end_dt,
            )
            continue
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")
        logger.debug(" -> Fetching %s (%s): %s to %s", name, season, start_str, end_str)
        pd_data = fetch_with_retries(pb.statcast_pitcher, start_str, end_str, pitcher_id)
        pd_data = filter_regular_season(pd_data)
        if pd_data is None:
            logger.error(
                " -> Error fetching Hist Statcast P %s (%s) season %s after retries.",
                name,
                pitcher_id,
                season,
            )
            fetch_failed = True
            continue
        if not pd_data.empty:
            try:
                pid_primitive = pitcher_id.item() if hasattr(pitcher_id, "item") else pitcher_id
                pd_data["pitcher_id"] = pid_primitive
                pd_data["season"] = season
                all_data.append(pd_data)
                logger.debug(" -> Fetched %d rows for %s (%s)", len(pd_data), name, season)
            except Exception as exc:
                logger.error(" -> Error adding identifiers for %s (%s): %s", name, season, exc)
                fetch_failed = True
    if fetch_failed:
        problematic_ids.add(pitcher_id)
    if not all_data:
        logger.debug(" -> No historical Statcast data found or fetched successfully for %s (%s).", name, pitcher_id)
        if not fetch_failed:
            checkpoint_manager.add_processed_pitcher(pitcher_id)
            logger.debug(" -> Marked P %s (%s) as processed (no data found/errors).", name, pitcher_id)
        return pd.DataFrame()
    try:
        combined_data = pd.concat(all_data, ignore_index=True)
        numeric_cols = ["release_speed", "release_spin_rate", "launch_speed", "launch_angle"]
        for col in numeric_cols:
            if col in combined_data.columns:
                combined_data[col] = pd.to_numeric(combined_data[col], errors="coerce")
        essential_cols = ["game_pk", "pitcher", "batter", "pitch_number"]
        if all(col in combined_data.columns for col in essential_cols):
            combined_data = combined_data.dropna(subset=essential_cols)
            combined_data = dedup_pitch_df(combined_data)
        else:
            logger.warning(
                "Missing essential columns in combined historical data for P %s (%s).", name, pitcher_id
            )
        logger.debug("Combined %d historical rows for P %s.", len(combined_data), name)
        if not fetch_failed:
            checkpoint_manager.add_processed_pitcher(pitcher_id)
            logger.debug(" -> Marked P %s (%s) as processed.", name, pitcher_id)
        return combined_data
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error combining or cleaning Hist Statcast for %s (%s): %s", name, pitcher_id, exc)
        problematic_ids.add(pitcher_id)
        return pd.DataFrame()
