from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Sequence

from src.utils import DBConnection


def deduplicate_table(db_path: Path, table: str, subset: Sequence[str] | None = None) -> None:
    """Remove duplicate rows from ``table`` in ``db_path``.

    If ``subset`` is provided, duplicates are identified using those
    columns. Otherwise exact row duplicates across all columns are
    removed.
    """
    with DBConnection(db_path) as conn:
        cur = conn.execute(f"PRAGMA table_info('{table}')")
        cols = [row[1] for row in cur.fetchall()]
        if not cols:
            raise ValueError(f"Table '{table}' does not exist or has no columns")

        col_list = ", ".join(f'"{c}"' for c in cols)
        tmp_table = f"{table}_dedup"
        if subset:
            subset_cols = ", ".join(f'"{c}"' for c in subset)
            query = (
                f"CREATE TABLE '{tmp_table}' AS "
                f"SELECT {col_list} FROM '{table}' GROUP BY {subset_cols}"
            )
        else:
            query = (
                f"CREATE TABLE '{tmp_table}' AS "
                f"SELECT DISTINCT {col_list} FROM '{table}'"
            )
        conn.execute(query)
        conn.execute(f"DROP TABLE '{table}'")
        conn.execute(f"ALTER TABLE '{tmp_table}' RENAME TO '{table}'")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove duplicate rows from a SQLite table")
    parser.add_argument("--db-path", type=Path, required=True, help="Path to SQLite database")
    parser.add_argument("--table", required=True, help="Table name to deduplicate")
    parser.add_argument(
        "--subset",
        nargs="+",
        default=None,
        help="Columns that define a duplicate row (defaults to all columns)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    deduplicate_table(args.db_path, args.table, args.subset)


if __name__ == "__main__":  # pragma: no cover
    main()
