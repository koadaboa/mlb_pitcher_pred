from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional

import httpx
import pandas as pd

from src.utils import DBConnection, setup_logger, table_exists
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
    """Return starting lineup rows from boxscore JSON with catcher info."""
    rows: List[Dict] = []
    teams = data.get("teams", {})
    for side in ("home", "away"):
        team_data = teams.get(side, {})
        team = team_data.get("team", {}).get("abbreviation")
        players = team_data.get("players", {})
        catcher_id: int | None = None
        for p in players.values():
            order = p.get("battingOrder")
            if not order:
                continue
            try:
                order_int = int(order)
            except (TypeError, ValueError):
                continue
            if order_int > 199:
                continue
            position = (
                p.get("position", {}).get("abbreviation")
                or p.get("position", {}).get("code")
                or p.get("gamePosition")
            )
            if position == "C" or position == "2":
                catcher_id = p.get("person", {}).get("id")
        # Now add lineup rows including the identified catcher_id
        for p in players.values():
            order = p.get("battingOrder")
            if not order:
                continue
            try:
                order_int = int(order)
            except (TypeError, ValueError):
                continue
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
                    "catcher_id": catcher_id,
                }
            )
    return rows


def build_starting_lineups(
    db_path: Path = DBConfig.PATH,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Query the MLB API for starting lineups of all games in ``mlb_boxscores``.

    Parameters
    ----------
    rebuild : bool, default False
        If ``True`` drop the existing ``game_starting_lineups`` table and rebuild
        it from scratch.
    """

    with DBConnection(db_path) as conn:
        games = pd.read_sql_query("SELECT game_pk FROM mlb_boxscores", conn)
        if not rebuild and table_exists(conn, "game_starting_lineups"):
            existing = pd.read_sql_query(
                "SELECT DISTINCT game_pk FROM game_starting_lineups", conn
            )
            processed = set(existing["game_pk"].astype(int))
        else:
            processed = set()

    rows: List[Dict] = []
    for game_pk in games["game_pk"].unique():
        if int(game_pk) in processed:
            continue
        data = fetch_boxscore(int(game_pk))
        if not data:
            continue
        rows.extend(parse_lineups(int(game_pk), data))

    if not rows:
        logger.info("No new starting lineups to process")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    with DBConnection(db_path) as conn:
        if rebuild or not table_exists(conn, "game_starting_lineups"):
            df.to_sql("game_starting_lineups", conn, index=False, if_exists="replace")
        else:
            df.to_sql("game_starting_lineups", conn, index=False, if_exists="append")
    return df


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch MLB starting lineups")
    parser.add_argument(
        "--db-path", type=Path, default=DBConfig.PATH, help="Path to SQLite DB"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop existing lineup table and refetch all games",
    )
    args = parser.parse_args(argv)

    try:
        df = build_starting_lineups(db_path=args.db_path, rebuild=args.rebuild)
        logger.info("Wrote %d starting lineup rows", len(df))
    except Exception as exc:
        logger.exception("Failed to build starting lineups: %s", exc)


if __name__ == "__main__":
    main()
