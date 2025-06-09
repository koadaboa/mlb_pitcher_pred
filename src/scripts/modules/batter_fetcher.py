"""Helpers for fetching batter statcast data."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pybaseball as pb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import logging

from src.utils import DBConnection
from src.config import DataConfig
from .fetch_utils import dedup_pitch_df
from .checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


def fetch_batter_single_date(
    target_date_obj: date,
    db_path: Path,
    fetch_with_retries,
    failed_batter_fetches: set,
) -> bool:
    """Fetch batter statcast data for a single date."""
    target_date_str = target_date_obj.strftime("%Y-%m-%d")
    target_season = target_date_obj.year
    data_exists = False
    try:
        with DBConnection(db_path) as conn:
            query = "SELECT COUNT(*) FROM statcast_batters WHERE DATE(game_date) = ?"
            cursor = conn.cursor()
            cursor.execute(query, (target_date_str,))
            count = cursor.fetchone()[0]
            if count > 0:
                data_exists = True
    except Exception as exc:
        logger.warning("DB check failed for batters on %s: %s. Will attempt fetch.", target_date_str, exc)
    if data_exists:
        logger.info("Skipping batter fetch: Data already exists for %s", target_date_str)
        return True
    logger.info("Fetching batter Statcast for single date: %s", target_date_str)
    pdata = fetch_with_retries(pb.statcast, start_dt=target_date_str, end_dt=target_date_str)
    if pdata is None:
        logger.error("Error fetching batter data for single date %s after retries.", target_date_str)
        failed_batter_fetches.add(("single_date", target_date_str, target_date_str))
        return False
    if pdata.empty:
        logger.info("No batter data found for %s.", target_date_str)
        return True
    try:
        pdata["season"] = target_season
        numeric_cols = ["release_speed", "launch_speed", "launch_angle", "woba_value"]
        for col in numeric_cols:
            if col in pdata.columns:
                pdata[col] = pd.to_numeric(pdata[col], errors="coerce")
        essential_cols = ["batter", "pitcher", "game_pk"]
        if all(col in pdata.columns for col in essential_cols):
            pdata = pdata.dropna(subset=essential_cols)
        else:
            logger.warning("Missing essential columns in fetched batter data for %s", target_date_str)
        if pdata.empty:
            logger.info("Batter data for %s was empty after cleaning.", target_date_str)
            return True
        rows_to_store = len(pdata)
        success = store_data_to_sql(pdata, "statcast_batters", db_path, if_exists="append")
        if success:
            logger.info("Stored %d batter rows for %s.", rows_to_store, target_date_str)
            return True
        logger.error("Failed to store batter data for %s.", target_date_str)
        failed_batter_fetches.add(("single_date", target_date_str, target_date_str))
        return False
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error processing/storing single date batter data for %s: %s", target_date_str, exc)
        failed_batter_fetches.add(("single_date", target_date_str, target_date_str))
        return False


def fetch_batter_historical(
    seasons: List[int],
    end_date_limit: date,
    db_path: Path,
    fetch_with_retries,
    checkpoint_manager: CheckpointManager,
    failed_batter_fetches: set,
    parallel: bool = False,
) -> bool:
    """Fetch batter statcast data across multiple seasons respecting checkpoints."""
    logger.info("Starting historical batter Statcast fetch up to %s", end_date_limit)
    total_stored_rows = 0
    overall_success = True
    seasons_to_process = [s for s in sorted(seasons) if s <= end_date_limit.year]
    if not seasons_to_process:
        logger.info("No relevant seasons for historical batter fetch based on end date limit.")
        return True
    for season in seasons_to_process:
        logger.info("--- Processing Batter Season: %s ---", season)
        season_end_limit = end_date_limit if season == end_date_limit.year else date(season, 11, 30)
        last_processed = checkpoint_manager.get_last_processed_batter_date(season)
        start_fetch_dt = date(season, 3, 1)
        if last_processed:
            try:
                last_dt = datetime.strptime(last_processed, "%Y-%m-%d").date()
                start_fetch_dt = last_dt + timedelta(days=1)
                logger.info("Resuming season %s batter fetch from %s", season, start_fetch_dt.strftime("%Y-%m-%d"))
            except ValueError:
                logger.warning("Invalid last processed date '%s' for season %s. Starting from beginning.", last_processed, season)
        if start_fetch_dt > season_end_limit:
            logger.info("Season %s already fully processed up to %s. Skipping.", season, season_end_limit)
            continue
        ranges_to_fetch = []
        current_chunk_start = start_fetch_dt
        chunk_days = getattr(DataConfig, "CHUNK_SIZE", 14)
        while current_chunk_start <= season_end_limit:
            current_chunk_end = min(current_chunk_start + timedelta(days=chunk_days - 1), season_end_limit)
            ranges_to_fetch.append((current_chunk_start.strftime("%Y-%m-%d"), current_chunk_end.strftime("%Y-%m-%d")))
            current_chunk_start = current_chunk_end + timedelta(days=1)
        if not ranges_to_fetch:
            logger.info("No new date ranges to fetch for season %s.", season)
            continue
        logger.info("Generated %d date ranges to fetch for season %s", len(ranges_to_fetch), season)
        processed_chunks_count = 0
        season_stored_rows = 0
        successful_end_dates: List[str] = []

        def process_chunk(start_str: str, end_str: str) -> Tuple[bool, str | None]:
            nonlocal season_stored_rows
            fetch_key = (season, start_str, end_str)
            logger.debug("Fetching hist batter: %s to %s", start_str, end_str)
            pdata = fetch_with_retries(pb.statcast, start_dt=start_str, end_dt=end_str)
            if pdata is None:
                logger.error(" -> Error fetching hist batter range %s-%s for season %s after retries.", start_str, end_str, season)
                failed_batter_fetches.add(fetch_key)
                return False, None
            if pdata.empty:
                logger.debug(" -> No data found for batter range %s-%s.", start_str, end_str)
                return True, end_str
            try:
                pdata["season"] = season
                numeric_cols = ["release_speed", "launch_speed", "launch_angle", "woba_value"]
                for col in numeric_cols:
                    if col in pdata.columns:
                        pdata[col] = pd.to_numeric(pdata[col], errors="coerce")
                essential_cols = ["batter", "pitcher", "game_pk"]
                if all(col in pdata.columns for col in essential_cols):
                    pdata = pdata.dropna(subset=essential_cols)
                else:
                    logger.warning("Missing essential columns in fetched batter data for %s-%s.", start_str, end_str)
                if pdata.empty:
                    logger.debug("Batter data for range %s-%s was empty after cleaning.", start_str, end_str)
                    return True, end_str
                rows_to_store = len(pdata)
                success = store_data_to_sql(pdata, "statcast_batters", db_path, if_exists="append")
                if success:
                    logger.debug(" -> Stored %d batter rows for range %s-%s.", rows_to_store, start_str, end_str)
                    season_stored_rows += rows_to_store
                    return True, end_str
                logger.error(" -> Failed to store batter data for range %s-%s.", start_str, end_str)
                failed_batter_fetches.add(fetch_key)
                return False, None
            except Exception as exc:
                logger.error(
                    " -> Error processing/storing hist batter range %s-%s: %s", start_str, end_str, exc, exc_info=True
                )
                failed_batter_fetches.add(fetch_key)
                return False, None

        if parallel:
            workers = min(DataConfig.MAX_WORKERS, os.cpu_count() or 1)
            logger.info("Using PARALLEL batter fetch for season %s (%s workers).", season, workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_range = {
                    executor.submit(process_chunk, start, end): (start, end) for start, end in ranges_to_fetch
                }
                progress_bar = tqdm(as_completed(future_to_range), total=len(ranges_to_fetch), desc=f"Batter Chunks S{season} (Parallel)")
                for future in progress_bar:
                    start, end = future_to_range[future]
                    try:
                        chunk_success, processed_end_date = future.result()
                        if chunk_success and processed_end_date:
                            successful_end_dates.append(processed_end_date)
                            processed_chunks_count += 1
                        elif not chunk_success:
                            overall_success = False
                        progress_bar.set_postfix_str(
                            f"Stored: {season_stored_rows}, Errors: {len(failed_batter_fetches)}"
                        )
                    except Exception as exc:
                        logger.error("Error processing result for batter range %s-%s: %s", start, end, exc, exc_info=True)
                        failed_batter_fetches.add((season, start, end))
                        overall_success = False
        else:
            logger.info("Using SEQUENTIAL batter fetch for season %s.", season)
            progress_bar = tqdm(ranges_to_fetch, desc=f"Batter Chunks S{season} (Sequential)")
            for start, end in progress_bar:
                try:
                    chunk_success, processed_end_date = process_chunk(start, end)
                    if chunk_success and processed_end_date:
                        successful_end_dates.append(processed_end_date)
                        processed_chunks_count += 1
                    elif not chunk_success:
                        overall_success = False
                    progress_bar.set_postfix_str(
                        f"Stored: {season_stored_rows}, Errors: {len(failed_batter_fetches)}"
                    )
                except Exception as exc:
                    logger.error("Critical error processing batter range %s-%s sequentially: %s", start, end, exc, exc_info=True)
                    failed_batter_fetches.add((season, start, end))
                    overall_success = False
        if successful_end_dates:
            try:
                max_successful_date_str = max(successful_end_dates)
                checkpoint_manager.update_last_processed_batter_date(season, max_successful_date_str)
                logger.info("Updated checkpoint for season %s to last successful date: %s", season, max_successful_date_str)
            except Exception as exc:
                logger.error("Failed to determine or update max successful date for season %s: %s", season, exc)
                overall_success = False
        elif ranges_to_fetch:
            logger.warning("No chunks successfully processed for season %s. Checkpoint not updated.", season)
            overall_success = False
        checkpoint_manager.save_overall_checkpoint()
        total_stored_rows += season_stored_rows
        if not overall_success:
            logger.error("Season %s completed with errors.", season)
        else:
            logger.info(
                "Season %s batter fetch phase complete. Processed %s/%s chunks. Stored %s new rows this season.",
                season,
                processed_chunks_count,
                len(ranges_to_fetch),
                season_stored_rows,
            )
    logger.info(
        "Historical batter fetch completed. Total stored rows across all seasons: %s.",
        total_stored_rows,
    )
    if not overall_success:
        logger.error("Historical batter fetching finished with errors in one or more seasons.")
    return overall_success

from .store_utils import store_data_to_sql  # noqa: E402
