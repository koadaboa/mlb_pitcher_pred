# checkpoint_manager.py
from __future__ import annotations
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from src.utils import ensure_dir, setup_logger
from src.config import LogConfig

logger = setup_logger("checkpoint_manager")

project_root = Path(__file__).resolve().parents[2]

class CheckpointManager:
    """Manage progress checkpoints for long running fetch jobs."""

    def __init__(self, checkpoint_dir: Path | str = project_root / "data" / ".checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        ensure_dir(self.checkpoint_dir)
        self.checkpoint_file = self.checkpoint_dir / "data_fetcher_progress.json"
        self.current_checkpoint: Dict[str, Any] = {}
        self.lock = threading.Lock()
        try:
            self.load_overall_checkpoint()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("CheckpointManager init error: %s. Init fresh.", exc)
            self._initialize_checkpoint()
            self.save_overall_checkpoint()

    def _initialize_checkpoint(self) -> None:
        self.current_checkpoint = {
            "processed_pitcher_ids": [],
            "last_processed_batter_date": {},
            "processed_mlb_api_dates": [],
            "last_update": datetime.now().isoformat(),
        }

    def _ensure_keys(self) -> None:
        defaults = {
            "processed_pitcher_ids": [],
            "last_processed_batter_date": {},
            "processed_mlb_api_dates": [],
            "last_update": datetime.now().isoformat(),
        }
        updated = False
        for key, default_val in defaults.items():
            current_val = self.current_checkpoint.get(key)
            correct_type = isinstance(current_val, type(default_val))
            if key not in self.current_checkpoint or current_val is None or not correct_type:
                if key != "last_update":
                    logger.warning("Initializing/Resetting checkpoint key '%s'", key)
                self.current_checkpoint[key] = default_val
                updated = True
        old_keys = [
            "pitcher_mapping_completed",
            "processed_seasons_batter_data",
            "team_batting_completed",
        ]
        for old_key in old_keys:
            if old_key in self.current_checkpoint:
                logger.warning("Removing obsolete checkpoint key: '%s'", old_key)
                del self.current_checkpoint[old_key]
                updated = True
        if updated:
            self.save_overall_checkpoint()

    def load_overall_checkpoint(self) -> None:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as fp:
                    self.current_checkpoint = json.load(fp)
                logger.info("Loaded checkpoint from %s", self.checkpoint_file)
            except json.JSONDecodeError as exc:
                logger.error("Failed to decode checkpoint JSON (%s)", exc)
                self._initialize_checkpoint()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed load checkpoint file %s (%s)", self.checkpoint_file, exc)
                self._initialize_checkpoint()
        else:
            logger.info("No checkpoint file found. Initializing new checkpoint.")
            self._initialize_checkpoint()
        self._ensure_keys()

    def save_overall_checkpoint(self) -> None:
        with self.lock:
            self.current_checkpoint["last_update"] = datetime.now().isoformat()
            temp_file = self.checkpoint_file.with_suffix(".tmp")
            try:
                for key in ["processed_pitcher_ids", "processed_mlb_api_dates"]:
                    current_list = self.current_checkpoint.get(key, [])
                    if isinstance(current_list, list):
                        try:
                            s_list = [item.item() if hasattr(item, "item") else item for item in current_list]
                            self.current_checkpoint[key] = sorted(list(set(s_list)))
                        except TypeError:
                            logger.warning("Could not sort checkpoint list '%s'", key)
                            self.current_checkpoint[key] = current_list
                with open(temp_file, "w") as fp:
                    json.dump(self.current_checkpoint, fp, indent=4)
                os.replace(temp_file, self.checkpoint_file)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to save checkpoint to %s: %s", self.checkpoint_file, exc)
                if temp_file.exists():
                    try:
                        os.remove(temp_file)
                    except OSError:
                        logger.error("Failed to remove temporary checkpoint file %s", temp_file)

    # -- Pitcher helpers
    def add_processed_pitcher(self, pitcher_id: int) -> None:
        with self.lock:
            p_list = self.current_checkpoint.setdefault("processed_pitcher_ids", [])
            item_id = pitcher_id.item() if hasattr(pitcher_id, "item") else pitcher_id
            if item_id not in p_list:
                p_list.append(item_id)

    def is_pitcher_processed(self, pitcher_id: int) -> bool:
        item_id = pitcher_id.item() if hasattr(pitcher_id, "item") else pitcher_id
        return item_id in self.current_checkpoint.get("processed_pitcher_ids", [])

    # -- Batter helpers
    def get_last_processed_batter_date(self, season: int) -> str | None:
        season_str = str(season)
        last_date_str = self.current_checkpoint.get("last_processed_batter_date", {}).get(season_str)
        if last_date_str:
            try:
                datetime.strptime(last_date_str, "%Y-%m-%d")
                return last_date_str
            except ValueError:
                logger.warning("Invalid date format '%s' in checkpoint", last_date_str)
                with self.lock:
                    if season_str in self.current_checkpoint.get("last_processed_batter_date", {}):
                        del self.current_checkpoint["last_processed_batter_date"][season_str]
        return None

    def update_last_processed_batter_date(self, season: int, date_str: str) -> None:
        with self.lock:
            season_str = str(season)
            date_map = self.current_checkpoint.setdefault("last_processed_batter_date", {})
            current_last = date_map.get(season_str)
            should_update = True
            if current_last:
                try:
                    cur_dt = datetime.strptime(current_last, "%Y-%m-%d").date()
                    new_dt = datetime.strptime(date_str, "%Y-%m-%d").date()
                    if new_dt <= cur_dt:
                        should_update = False
                except ValueError:
                    should_update = True
            if should_update:
                date_map[season_str] = date_str

    # -- MLB API helpers
    def get_last_processed_mlb_api_date(self) -> str | None:
        processed_list = self.current_checkpoint.get("processed_mlb_api_dates", [])
        if not processed_list:
            return None
        last_date_str = processed_list[-1]
        try:
            datetime.strptime(last_date_str, "%Y-%m-%d")
            return last_date_str
        except (ValueError, TypeError):
            for dt_str in reversed(processed_list[:-1]):
                try:
                    datetime.strptime(dt_str, "%Y-%m-%d")
                    return dt_str
                except (ValueError, TypeError):
                    continue
            logger.warning("No valid dates found in processed_mlb_api_dates")
            return None

    def add_processed_mlb_api_date(self, date_str: str) -> None:
        with self.lock:
            processed_list = self.current_checkpoint.setdefault("processed_mlb_api_dates", [])
            if date_str not in processed_list:
                processed_list.append(date_str)
                try:
                    self.current_checkpoint["processed_mlb_api_dates"] = sorted(processed_list)
                except TypeError:
                    logger.warning("Could not sort processed_mlb_api_dates list")

    def is_mlb_api_date_processed(self, date_str: str) -> bool:
        if not isinstance(date_str, str):
            return False
        return date_str in self.current_checkpoint.get("processed_mlb_api_dates", [])
