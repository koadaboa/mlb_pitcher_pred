from __future__ import annotations

import pandas as pd
from pathlib import Path
from src.utils import DBConnection
from src.config import DBConfig


def validate_unique_starters(
    db_path: Path = DBConfig.PATH,
    table: str = "game_level_starting_pitchers",
) -> pd.DataFrame:
    """Return DataFrame of duplicate starters if any are found."""
    with DBConnection(db_path) as conn:
        query = (
            f"SELECT game_pk, pitching_team, COUNT(*) as cnt FROM {table} "
            "GROUP BY game_pk, pitching_team HAVING COUNT(*) > 1"
        )
        dup_df = pd.read_sql_query(query, conn)
    if dup_df.empty:
        return dup_df
    raise ValueError(
        f"Found multiple starters for {len(dup_df)} game/team combos",
    )


def main() -> None:
    try:
        dup_df = validate_unique_starters()
        if dup_df.empty:
            print("All teams have exactly one starting pitcher per game.")
    except ValueError as exc:
        print(exc)


if __name__ == "__main__":
    main()
