import sqlite3
import pandas as pd
import pytest

from src.data.validate_starting_pitchers import validate_unique_starters


def test_validate_unique_starters(tmp_path):
    db_path = tmp_path / "test.db"
    with sqlite3.connect(db_path) as conn:
        df = pd.DataFrame({
            "game_pk": [1, 1],
            "pitching_team": ["A", "A"],
        })
        df.to_sql("game_level_starting_pitchers", conn, index=False)
    with pytest.raises(ValueError):
        validate_unique_starters(db_path=db_path)

    # Replace with unique rows
    with sqlite3.connect(db_path) as conn:
        df = pd.DataFrame({
            "game_pk": [1, 2],
            "pitching_team": ["A", "B"],
        })
        df.to_sql("game_level_starting_pitchers", conn, index=False, if_exists="replace")

    validate_unique_starters(db_path=db_path)
