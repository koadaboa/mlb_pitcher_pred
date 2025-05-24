from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import DBConnection, setup_logger
from src.config import DBConfig, LogConfig

logger = setup_logger(
    "create_team_batting",
    LogConfig.LOG_DIR / "create_team_batting.log",
)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    return (values[mask] * weights[mask]).sum() / weights[mask].sum()


def aggregate_from_batters(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["game_pk", "pitching_team", "opponent_team"]
    for (game_pk, pitching_team, opponent_team), g in df.groupby(group_cols):
        plate_appearances = g["plate_appearances"].sum()
        at_bats = g["at_bats"].sum()
        pitches = g["pitches"].sum()
        swings = g["swings"].sum()
        whiffs = g["whiffs"].sum()
        called_strike_total = (g["called_strike_rate"].fillna(0) * g["pitches"]).sum()
        strikeouts = g["strikeouts"].sum()
        hits = g["hits"].sum()
        singles = g["singles"].sum()
        doubles = g["doubles"].sum()
        triples = g["triples"].sum()
        home_runs = g["home_runs"].sum()
        walks = g["walks"].sum()
        hbp = g["hbp"].sum()

        row = {
            "game_pk": game_pk,
            "pitching_team": pitching_team,
            "opponent_team": opponent_team,
            "bat_plate_appearances": plate_appearances,
            "bat_at_bats": at_bats,
            "bat_pitches": pitches,
            "bat_swings": swings,
            "bat_whiffs": whiffs,
            "bat_whiff_rate": whiffs / swings if swings else np.nan,
            "bat_called_strike_rate": called_strike_total / pitches if pitches else np.nan,
            "bat_strikeouts": strikeouts,
            "bat_strikeout_rate": strikeouts / plate_appearances if plate_appearances else np.nan,
            "bat_strikeout_rate_behind": weighted_mean(g["strikeout_rate_behind"], g["plate_appearances"]),
            "bat_strikeout_rate_ahead": weighted_mean(g["strikeout_rate_ahead"], g["plate_appearances"]),
            "bat_hits": hits,
            "bat_singles": singles,
            "bat_doubles": doubles,
            "bat_triples": triples,
            "bat_home_runs": home_runs,
            "bat_walks": walks,
            "bat_hbp": hbp,
        }

        row["bat_avg"] = hits / at_bats if at_bats else np.nan
        row["bat_obp"] = (hits + walks + hbp) / plate_appearances if plate_appearances else np.nan
        row["bat_slugging"] = (
            singles + 2 * doubles + 3 * triples + 4 * home_runs
        ) / at_bats if at_bats else np.nan
        row["bat_ops"] = (
            row["bat_obp"] + row["bat_slugging"]
            if pd.notna(row["bat_obp"]) and pd.notna(row["bat_slugging"])
            else np.nan
        )
        row["bat_woba"] = weighted_mean(g["woba"], g["plate_appearances"])

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_team_batting(db_path: Path = DBConfig.PATH) -> pd.DataFrame:
    with DBConnection(db_path) as conn:
        batter_df = pd.read_sql_query("SELECT * FROM game_level_batters_vs_starters", conn)
    if batter_df.empty:
        logger.warning("No rows found in game_level_batters_vs_starters")
        team_df = pd.DataFrame()
    else:
        team_df = aggregate_from_batters(batter_df)
    with DBConnection(db_path) as conn:
        team_df.to_sql(
            "game_level_team_batting",
            conn,
            index=False,
            if_exists="replace",
        )
    return team_df


def main() -> None:
    try:
        df = aggregate_team_batting()
        logger.info("Aggregated %d team batting rows", len(df))
    except Exception as exc:
        logger.exception("Failed to create team batting table: %s", exc)


if __name__ == "__main__":
    main()
