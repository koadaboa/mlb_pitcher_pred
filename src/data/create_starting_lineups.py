from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

import httpx
import pandas as pd

from src.utils import DBConnection, setup_logger
from src.config import DBConfig, LogConfig

logger = setup_logger(
    "create_starting_lineups",
    LogConfig.LOG_DIR / "create_starting_lineups.log",
)

API_BASE = "https://statsapi.mlb.com/api/v1"


def fetch_boxscore(game_pk: int) -> Optional[dict]:
    """Fetch boxscore JSON for a single game."""
    url = f"{API_BASE}/game/{game_pk}/boxscore"
    try:
        with httpx.Client(timeout=20) as client:
            resp = client.get(url)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch boxscore for %s: %s", game_pk, exc)
        return None


def parse_lineups(game_pk: int, data: dict) -> List[Dict]:
    """Return starting lineup rows from boxscore JSON."""
    rows: List[Dict] = []
    teams = data.get("teams", {})
    for side in ("home", "away"):
        team_data = teams.get(side, {})
        team = team_data.get("team", {}).get("abbreviation")
        players = team_data.get("players", {})
        for p in players.values():
            order = p.get("battingOrder")
            if not order:
                continue
            try:
                order_int = int(order)
            except (TypeError, ValueError):
                continue
            # Only keep the first 9 lineup spots
            if order_int > 199:
                continue
            batter_id = p.get("person", {}).get("id")
            stand = p.get("batSide", {}).get("code")
            rows.append(
                {
                    "game_pk": game_pk,
                    "team": team,
                    "batter_id": batter_id,
                    "batting_order": order_int,
                    "stand": stand,
                }
            )
    return rows


def build_starting_lineups(db_path: Path = DBConfig.PATH) -> pd.DataFrame:
    """Query the MLB API for starting lineups of all games in ``mlb_boxscores``."""
    with DBConnection(db_path) as conn:
        games = pd.read_sql_query("SELECT game_pk FROM mlb_boxscores", conn)
    rows: List[Dict] = []
    for game_pk in games["game_pk"].unique():
        data = fetch_boxscore(int(game_pk))
        if not data:
            continue
        rows.extend(parse_lineups(int(game_pk), data))
    df = pd.DataFrame(rows)
    with DBConnection(db_path) as conn:
        df.to_sql("game_starting_lineups", conn, index=False, if_exists="replace")
    return df


def main() -> None:
    try:
        df = build_starting_lineups()
        logger.info("Wrote %d starting lineup rows", len(df))
    except Exception as exc:
        logger.exception("Failed to build starting lineups: %s", exc)


if __name__ == "__main__":
    main()
