"""Utility functions shared by fetcher helpers."""
from __future__ import annotations

import pandas as pd
import logging

UNIQUE_PITCH_COLS = ["game_pk", "pitcher", "inning", "batter", "pitch_number"]

logger = logging.getLogger(__name__)


def dedup_pitch_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate pitch rows based on key columns."""
    before = len(df)
    df = df.drop_duplicates(subset=UNIQUE_PITCH_COLS)
    dropped = before - len(df)
    if dropped > 0:
        logger.debug("Removed %d duplicate pitch rows", dropped)
    return df


def filter_regular_season(df: pd.DataFrame) -> pd.DataFrame:
    """Return only regular season rows if ``game_type`` column is present."""
    if "game_type" in df.columns:
        before = len(df)
        df = df[df["game_type"] == "R"]
        removed = before - len(df)
        if removed > 0:
            logger.debug("Filtered %d non-regular season rows", removed)
    return df
