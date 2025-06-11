from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional, Union
import logging
import re
from datetime import datetime
import pandas as pd

from src.config import DBConfig

class DBConnection:
    """Simple context manager for SQLite connections."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else Path(DBConfig.PATH)
        # ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> sqlite3.Connection:
        self.conn = sqlite3.connect(str(self.db_path))
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.conn:
            if exc_type:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.conn.close()
            self.conn = None


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure that the directory exists and return a ``Path`` object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(
    name: str, log_file: Optional[Union[str, Path]] = None,
    level: int = logging.WARNING,
) -> logging.Logger:
    """Create and return a console/file logger with a standard format.

    The level defaults to ``WARNING`` but can be overridden via the
    ``LOG_LEVEL`` environment variable.
    """
    env_level = os.getenv("LOG_LEVEL")
    if env_level:
        level = getattr(logging, env_level.upper(), level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file and not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(Path(log_file)) for h in logger.handlers):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def find_latest_file(directory: Union[str, Path], pattern: str) -> Optional[Path]:
    """Return the most recent file in ``directory`` matching ``pattern``.

    The function first looks for timestamps in the filename formatted as
    ``_YYYYMMDD_HHMMSS``. If none are found it falls back to modification time.
    """
    try:
        search_dir = Path(directory)
        if not search_dir.is_dir():
            logging.error("Directory not found: %s", search_dir)
            return None

        files = list(search_dir.glob(pattern))
        if not files:
            logging.info("No files found matching pattern '%s' in %s", pattern, search_dir)
            return None

        latest = None
        latest_ts = 0.0
        ts_re = re.compile(r"_(\d{8}_\d{6})\.")
        for f in files:
            m = ts_re.search(f.name)
            if m:
                try:
                    ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp()
                    if ts > latest_ts:
                        latest_ts = ts
                        latest = f
                except ValueError:
                    logging.warning("Could not parse timestamp from %s", f.name)

        if not latest:
            latest = max(files, key=lambda x: x.stat().st_mtime)

        return latest
    except Exception as exc:
        logging.error("Unexpected error in find_latest_file: %s", exc, exc_info=True)
        return None


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Return ``True`` if ``table`` exists in the connected SQLite database."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    return cur.fetchone() is not None


def get_latest_date(
    conn: sqlite3.Connection, table: str, date_col: str = "game_date"
) -> Optional[pd.Timestamp]:
    """Return the most recent ``date_col`` value from ``table`` if it exists."""
    if not table_exists(conn, table):
        return None
    cur = conn.execute(f"SELECT MAX({date_col}) FROM {table}")
    row = cur.fetchone()
    if row and row[0] is not None:
        return pd.to_datetime(row[0])
    return None


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove pandas merge suffixes and resolve duplicate columns.

    Repeated merges can produce columns with nested ``_x``/``_y`` suffixes
    (e.g. ``foo_x_y``). This function consolidates such variants by stripping the
    suffixes and merging values.
    """

    import re

    pattern = re.compile(r"(?:_x|_y)+$")
    groups: dict[str, list[str]] = {}
    for col in list(df.columns):
        base = re.sub(pattern, "", col)
        groups.setdefault(base, []).append(col)

    for base, cols in groups.items():
        if len(cols) == 1:
            col = cols[0]
            if col != base:
                df = df.rename(columns={col: base})
            continue

        main = base if base in cols else cols[0]
        for col in cols:
            if col == main:
                continue
            df[main] = df[main].combine_first(df[col])
            df = df.drop(columns=[col])
        if main != base:
            df = df.rename(columns={main: base})

    return df


def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *args,
    **kwargs,
) -> pd.DataFrame:
    """Merge two ``DataFrame`` objects and remove any duplicate columns."""

    left = deduplicate_columns(left)
    right = deduplicate_columns(right)
    merged = left.merge(right, *args, **kwargs)
    merged = deduplicate_columns(merged)
    return merged


