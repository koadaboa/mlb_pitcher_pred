import logging
from pathlib import Path
import pandas as pd

from src.utils import DBConnection, setup_logger
from src.config import DBConfig, LogConfig, STATCAST_PITCHERS_TABLE

logger = setup_logger(
    "check_statcast_duplicates",
    LogConfig.LOG_DIR / "check_statcast_duplicates.log",
)


def main(db_path: Path = DBConfig.PATH) -> None:
    """Check for duplicate pitch rows in the statcast_pitchers table."""
    with DBConnection(db_path) as conn:
        query = f"""
        SELECT
            game_pk,
            pitcher,
            inning,
            batter,
            pitch_number,
            COUNT(*) AS dup_count
        FROM {STATCAST_PITCHERS_TABLE}
        GROUP BY game_pk, pitcher, inning, batter, pitch_number
        HAVING dup_count > 1
        """
        df = pd.read_sql_query(query, conn)
        if df.empty:
            logger.info("No duplicates found in %s", STATCAST_PITCHERS_TABLE)
            return
        out_csv = Path("statcast_pitcher_duplicates.csv")
        df.to_csv(out_csv, index=False)
        logger.info("Found %d duplicate rows; results saved to %s", len(df), out_csv)


if __name__ == "__main__":
    main()
