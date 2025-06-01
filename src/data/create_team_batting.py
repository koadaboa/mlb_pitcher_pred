from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import DBConnection, setup_logger, safe_merge
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
            "bat_called_strike_rate": (
                called_strike_total / pitches if pitches else np.nan
            ),
            "bat_strikeouts": strikeouts,
            "bat_strikeout_rate": (
                strikeouts / plate_appearances if plate_appearances else np.nan
            ),
            "bat_strikeout_rate_behind": weighted_mean(
                g["strikeout_rate_behind"], g["plate_appearances"]
            ),
            "bat_strikeout_rate_ahead": weighted_mean(
                g["strikeout_rate_ahead"], g["plate_appearances"]
            ),
            "bat_hits": hits,
            "bat_singles": singles,
            "bat_doubles": doubles,
            "bat_triples": triples,
            "bat_home_runs": home_runs,
            "bat_walks": walks,
            "bat_hbp": hbp,
        }

        row["bat_avg"] = hits / at_bats if at_bats else np.nan
        row["bat_obp"] = (
            (hits + walks + hbp) / plate_appearances if plate_appearances else np.nan
        )
        row["bat_slugging"] = (
            (singles + 2 * doubles + 3 * triples + 4 * home_runs) / at_bats
            if at_bats
            else np.nan
        )
        row["bat_ops"] = (
            row["bat_obp"] + row["bat_slugging"]
            if pd.notna(row["bat_obp"]) and pd.notna(row["bat_slugging"])
            else np.nan
        )
        row["bat_woba"] = weighted_mean(g["woba"], g["plate_appearances"])

        # compute left/right batter splits
        for hand, prefix in [("L", "L"), ("R", "R")]:
            sub = g[g["stand"] == hand]
            pa = sub["plate_appearances"].sum()
            so = sub["strikeouts"].sum()
            hits_h = sub["hits"].sum()
            singles_h = sub["singles"].sum()
            doubles_h = sub["doubles"].sum()
            triples_h = sub["triples"].sum()
            hr_h = sub["home_runs"].sum()
            walks_h = sub["walks"].sum()
            hbp_h = sub["hbp"].sum()
            at_bats_h = sub["at_bats"].sum()
            obp_h = (hits_h + walks_h + hbp_h) / pa if pa else np.nan
            slg_h = (
                (singles_h + 2 * doubles_h + 3 * triples_h + 4 * hr_h) / at_bats_h
                if at_bats_h
                else np.nan
            )
            ops_h = obp_h + slg_h if pd.notna(obp_h) and pd.notna(slg_h) else np.nan
            row[f"bat_{prefix}_plate_appearances"] = pa
            row[f"bat_{prefix}_strikeout_rate"] = so / pa if pa else np.nan
            row[f"bat_{prefix}_ops"] = ops_h

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_team_batting(db_path: Path = DBConfig.PATH) -> pd.DataFrame:
    with DBConnection(db_path) as conn:
        batter_df = pd.read_sql_query(
            "SELECT * FROM game_level_batters_vs_starters", conn
        )
        starter_df = pd.read_sql_query(
            "SELECT game_pk, pitcher_hand FROM game_level_starting_pitchers",
            conn,
        )
    if batter_df.empty:
        logger.warning("No rows found in game_level_batters_vs_starters")
        team_df = pd.DataFrame()
    else:
        team_df = aggregate_from_batters(batter_df)
        team_df = safe_merge(team_df, starter_df, on="game_pk", how="left")
        team_df["bat_ops_vs_LHP"] = np.where(
            team_df["pitcher_hand"] == "L", team_df["bat_ops"], np.nan
        )
        team_df["bat_ops_vs_RHP"] = np.where(
            team_df["pitcher_hand"] == "R", team_df["bat_ops"], np.nan
        )
        team_df["bat_k_rate_vs_LHP"] = np.where(
            team_df["pitcher_hand"] == "L", team_df["bat_strikeout_rate"], np.nan
        )
        team_df["bat_k_rate_vs_RHP"] = np.where(
            team_df["pitcher_hand"] == "R", team_df["bat_strikeout_rate"], np.nan
        )
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
