import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

import logging

from src.utils import DBConnection, setup_logger
from src.config import (
    DBConfig,
    LogConfig,
    STATCAST_BATTERS_TABLE,
    STATCAST_PITCHERS_TABLE,
)

BATTERS_VS_STARTERS_TABLE = "game_level_batters_vs_starters"

logger = setup_logger(
    "create_batter_vs_starter_table",
    LogConfig.LOG_DIR / "create_batter_vs_starter_table.log",
)


HIT_EVENTS = {"single", "double", "triple", "home_run"}
NON_AB_EVENTS = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interference", "intent_walk"}


def get_starting_pitchers(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return DataFrame of starting pitcher IDs per game and team."""
    query = f"""
    WITH pitch_team AS (
        SELECT ROWID AS rid,
               game_pk,
               CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END AS pitching_team,
               CASE WHEN inning_topbot = 'Top' THEN away_team ELSE home_team END AS opponent_team,
               pitcher_id
        FROM {STATCAST_PITCHERS_TABLE}
    ), first_pitch AS (
        SELECT game_pk, pitching_team, opponent_team, MIN(rid) AS min_rid
        FROM pitch_team
        GROUP BY game_pk, pitching_team, opponent_team
    ), pitch_counts AS (
        SELECT game_pk, pitcher_id, COUNT(*) AS pitch_count
        FROM {STATCAST_PITCHERS_TABLE}
        GROUP BY game_pk, pitcher_id

    )
    SELECT pt.game_pk, pt.pitching_team, pt.opponent_team, pt.pitcher_id
    FROM pitch_team pt
    JOIN first_pitch fp
      ON pt.game_pk = fp.game_pk
     AND pt.pitching_team = fp.pitching_team
     AND pt.opponent_team = fp.opponent_team
     AND pt.rid = fp.min_rid
    """
    return pd.read_sql_query(query, conn)


def compute_batter_features(df: pd.DataFrame) -> dict:
    df = df.sort_values(["at_bat_number", "pitch_number"]).reset_index(drop=True)
    num_pitches = len(df)
    pa_df = df.groupby("at_bat_number").tail(1)
    plate_appearances = len(pa_df)
    at_bats_mask = ~pa_df["events"].isin(NON_AB_EVENTS)
    at_bats = pa_df[at_bats_mask].shape[0]

    hits = pa_df["events"].isin(HIT_EVENTS).sum()
    singles = pa_df["events"].eq("single").sum()
    doubles = pa_df["events"].eq("double").sum()
    triples = pa_df["events"].eq("triple").sum()
    homers = pa_df["events"].eq("home_run").sum()
    walks = pa_df["events"].eq("walk").sum()
    hbp = pa_df["events"].eq("hit_by_pitch").sum()
    strikeouts = pa_df["events"].str.contains("strikeout", case=False, na=False).sum()

    swings = df["description"].str.contains("swing", case=False, na=False).sum()
    swings += df["description"].str.contains("foul", case=False, na=False).sum()
    swings += df["description"].str.contains("hit_into_play", case=False, na=False).sum()
    whiffs = df["description"].str.contains("swinging_strike", case=False, na=False).sum()
    called_strikes = df["description"].str.contains("called_strike", case=False, na=False).sum()

    behind_pa = 0
    behind_k = 0
    ahead_pa = 0
    ahead_k = 0
    for _, grp in df.groupby("at_bat_number"):
        behind = (grp["strikes"] > grp["balls"]).any()
        ahead = (grp["balls"] > grp["strikes"]).any()
        final_event = grp.iloc[-1]["events"]
        if behind:
            behind_pa += 1
            if isinstance(final_event, str) and "strikeout" in final_event.lower():
                behind_k += 1
        if ahead:
            ahead_pa += 1
            if isinstance(final_event, str) and "strikeout" in final_event.lower():
                ahead_k += 1

    woba_denom_sum = df["woba_denom"].sum()
    woba = df["woba_value"].sum() / woba_denom_sum if woba_denom_sum else np.nan

    total_bases = singles + 2 * doubles + 3 * triples + 4 * homers
    avg = hits / at_bats if at_bats else np.nan
    obp = (hits + walks + hbp) / plate_appearances if plate_appearances else np.nan
    slg = total_bases / at_bats if at_bats else np.nan
    ops = obp + slg if not np.isnan(obp) and not np.isnan(slg) else np.nan

    return {
        "plate_appearances": plate_appearances,
        "at_bats": at_bats,
        "pitches": num_pitches,
        "swings": swings,
        "whiffs": whiffs,
        "whiff_rate": whiffs / swings if swings else np.nan,
        "called_strike_rate": called_strikes / num_pitches if num_pitches else np.nan,
        "strikeouts": strikeouts,
        "strikeout_rate": strikeouts / plate_appearances if plate_appearances else np.nan,
        "strikeout_rate_behind": behind_k / behind_pa if behind_pa else np.nan,
        "strikeout_rate_ahead": ahead_k / ahead_pa if ahead_pa else np.nan,
        "hits": hits,
        "singles": singles,
        "doubles": doubles,
        "triples": triples,
        "home_runs": homers,
        "walks": walks,
        "hbp": hbp,
        "avg": avg,
        "obp": obp,
        "slugging": slg,
        "ops": ops,
        "woba": woba,
    }


def main(db_path: Path = DBConfig.PATH) -> None:
    """Build or replace the batter vs starter aggregation table."""
    logger.info("Starting batter vs starter aggregation")
    with DBConnection(db_path) as conn:
        starters = get_starting_pitchers(conn)
        logger.info("Found %d starting pitchers", len(starters))
        rows = []
        for _, s in starters.iterrows():
            logger.info(
                "Processing game %s pitcher %s for team %s",
                s.game_pk,
                s.pitcher_id,
                s.pitching_team,
            )

            df = pd.read_sql_query(
                f"SELECT batter, at_bat_number, pitch_number, events, description, balls, strikes, woba_value, woba_denom FROM {STATCAST_BATTERS_TABLE} WHERE game_pk=? AND pitcher=?",
                conn,
                params=(s.game_pk, s.pitcher_id),
            )
            if df.empty:

                logger.debug(
                    "No pitch data for game %s pitcher %s", s.game_pk, s.pitcher_id
                )
                continue
            for batter_id, bdf in df.groupby("batter"):
                feats = compute_batter_features(bdf)
                feats.update(
                    {
                        "game_pk": s.game_pk,
                        "batter_id": batter_id,
                        "pitcher_id": s.pitcher_id,
                        "pitching_team": s.pitching_team,
                        "opponent_team": s.opponent_team,
                    }
                )
                rows.append(feats)
        if rows:
            out_df = pd.DataFrame(rows)
            out_df.to_sql(
                BATTERS_VS_STARTERS_TABLE, conn, if_exists="replace", index=False
            )
            logger.info(
                "Wrote %d rows to table %s",
                len(out_df),
                BATTERS_VS_STARTERS_TABLE,
            )
        else:
            logger.warning("No rows generated for %s", BATTERS_VS_STARTERS_TABLE)
f.to_sql(BATTERS_VS_STARTERS_TABLE, conn, if_exists="replace", index=False)



if __name__ == "__main__":
    main()
