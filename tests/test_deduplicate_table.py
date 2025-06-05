import sqlite3
from pathlib import Path
import pandas as pd

from src.scripts import deduplicate_table as dedup


def test_deduplicate_table(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    df = pd.DataFrame(
        {
            "game_pk": [1, 1, 2],
            "pitcher": [100, 100, 200],
            "inning": [1, 1, 1],
            "batter": [10, 10, 20],
            "pitch_number": [1, 1, 1],
            "extra": [0, 0, 1],
        }
    )
    with sqlite3.connect(db_path) as conn:
        df.to_sql("statcast_pitchers", conn, index=False)

    subset = ["game_pk", "pitcher", "inning", "batter", "pitch_number"]
    dedup.deduplicate_table(db_path, "statcast_pitchers", subset=subset)

    with sqlite3.connect(db_path) as conn:
        result = pd.read_sql_query("SELECT * FROM statcast_pitchers", conn)

    assert len(result) == 2
    assert set(result["game_pk"]) == {1, 2}
