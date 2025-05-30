from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import httpx
import pandas as pd

from src.utils import DBConnection, setup_logger, ensure_dir
from src.config import DBConfig, LogConfig

API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
MARKET = "player_strikeouts"

logger = setup_logger("fetch_strikeout_odds", LogConfig.LOG_DIR / "fetch_strikeout_odds.log")


def fetch_odds(api_key: str) -> List[Dict[str, Any]]:
    """Fetch strikeout prop odds from the Odds API."""
    params = {
        "apiKey": api_key,
        "markets": MARKET,
        "regions": "us",
        "oddsFormat": "american",
    }
    logger.info("Requesting odds from The Odds API")
    with httpx.Client(timeout=20) as client:
        resp = client.get(API_URL, params=params)
        resp.raise_for_status()
        return resp.json()


def odds_to_df(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert Odds API JSON to a tidy DataFrame."""
    rows = []
    for event in data:
        event_id = event.get("id")
        commence = event.get("commence_time")
        for book in event.get("bookmakers", []):
            book_title = book.get("title")
            last_update = book.get("last_update")
            for market in book.get("markets", []):
                if market.get("key") != MARKET:
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price")
                    line = outcome.get("point") or outcome.get("line")
                    player = name
                    side = "unknown"
                    if " Over " in name:
                        player, _ = name.split(" Over ", 1)
                        side = "over"
                    elif " Under " in name:
                        player, _ = name.split(" Under ", 1)
                        side = "under"
                    rows.append({
                        "event_id": event_id,
                        "commence_time": commence,
                        "bookmaker": book_title,
                        "player": player.strip(),
                        "line": line,
                        "side": side,
                        "odds": price,
                        "last_update": last_update,
                        "inserted_at": datetime.utcnow().isoformat(),
                    })
    return pd.DataFrame(rows)


def save_odds(df: pd.DataFrame, db_path: Path, table: str = "strikeout_prop_odds") -> None:
    """Append odds DataFrame to SQLite."""
    if df.empty:
        logger.warning("No odds data to save")
        return
    with DBConnection(db_path) as conn:
        df.to_sql(table, conn, if_exists="append", index=False)
    logger.info("Saved %d rows to %s", len(df), table)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch MLB pitcher strikeout odds")
    parser.add_argument("--api-key", type=str, default=os.getenv("ODDS_API_KEY"), help="Odds API key")
    parser.add_argument("--db-path", type=Path, default=DBConfig.PATH, help="SQLite database path")
    parser.add_argument("--table", type=str, default="strikeout_prop_odds", help="Destination table name")
    args = parser.parse_args(argv)

    if not args.api_key:
        parser.error("API key required via --api-key or ODDS_API_KEY env var")

    ensure_dir(Path(args.db_path).parent)

    data = fetch_odds(args.api_key)
    df = odds_to_df(data)
    save_odds(df, args.db_path, args.table)


if __name__ == "__main__":
    main()
