from __future__ import annotations

import argparse
from pathlib import Path

from src.data.create_starting_pitcher_table import aggregate_to_game_level as build_starting_pitchers
from src.data.create_batters_vs_starters import aggregate_to_game_level as build_batters_vs_starters
from src.data.create_team_batting import aggregate_team_batting
from src.data.create_matchup_details_table import build_matchup_table
from src.data.create_catcher_defense import build_catcher_defense_metrics
from src.data.create_starting_lineups import build_starting_lineups
from src.data.create_pitcher_vs_team_stats import build_matchup_stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run initial aggregation scripts to build base tables"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and rebuild tables where supported",
    )
    args = parser.parse_args(argv)

    kwargs = {"db_path": args.db_path} if args.db_path else {}

    build_starting_pitchers(**kwargs)
    build_batters_vs_starters(**kwargs)
    aggregate_team_batting(**kwargs)
    build_matchup_stats(**kwargs)
    build_matchup_table(**kwargs)
    build_starting_lineups(db_path=args.db_path, rebuild=args.rebuild)
    build_catcher_defense_metrics(db_path=args.db_path, rebuild=args.rebuild)


if __name__ == "__main__":  # pragma: no cover
    main()
