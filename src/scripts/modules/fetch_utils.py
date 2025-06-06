"""Utility functions shared by fetcher helpers."""
from __future__ import annotations

import pandas as pd
import logging

# Known first regular-season dates per year
REG_SEASON_START = {
    2016: "2016-04-03",
    2017: "2017-04-02",
    2018: "2018-03-29",
    2019: "2019-03-20",
    2020: "2020-07-23",
    2021: "2021-04-01",
    2022: "2022-04-07",
    2023: "2023-03-30",
    2024: "2024-03-28",
    2025: "2025-03-27",
}
DEFAULT_START_MONTH = 3
DEFAULT_START_DAY = 25

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


def _season_start_date(year: int) -> pd.Timestamp:
    start = REG_SEASON_START.get(year)
    if start:
        return pd.to_datetime(start)
    return pd.Timestamp(year=year, month=DEFAULT_START_MONTH, day=DEFAULT_START_DAY)


def filter_regular_season(df: pd.DataFrame) -> pd.DataFrame:
    """Return only regular season rows."""
    if df is None or df.empty:
        return df

    df = df.copy()

    if "game_type" in df.columns:
        before = len(df)
        df = df[df["game_type"] == "R"]
        removed = before - len(df)
        if removed > 0:
            logger.debug("Filtered %d non-regular season rows by game_type", removed)

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        mask = []
        for dt in df["game_date"]:
            if pd.isna(dt):
                mask.append(False)
                continue
            start_date = _season_start_date(dt.year)
            mask.append(dt >= start_date)
        before = len(df)
        df = df[pd.Series(mask, index=df.index)]
        removed = before - len(df)
        if removed > 0:
            logger.debug("Filtered %d rows prior to regular season start", removed)

    return df
