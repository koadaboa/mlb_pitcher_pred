import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

from src.utils import DBConnection, setup_logger
from src.config import (
    DBConfig,
    LogConfig,
    STATCAST_STARTING_PITCHERS_TABLE,
)

BATTERS_VS_STARTERS_TABLE = "game_level_batters_vs_starters"
STARTERS_TABLE = STATCAST_STARTING_PITCHERS_TABLE
TEAM_BATTING_TABLE = "game_level_team_batting"
MATCHUP_TABLE = "game_level_matchup_stats"

logger = setup_logger(
    "create_pitcher_vs_team_stats",
    LogConfig.LOG_DIR / "create_pitcher_vs_team_stats.log",
)

def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    weights = weights.fillna(0)
    if weights.sum() == 0:
        return np.nan
    return np.average(values.fillna(0), weights=weights)

def compute_team_batting_features(df: pd.DataFrame) -> dict:
    pa = df["plate_appearances"].sum()
    ab = df["at_bats"].sum()
    pitches = df["pitches"].sum()
    swings = df["swings"].sum()
    whiffs = df["whiffs"].sum()
    strikeouts = df["strikeouts"].sum()
    hits = df["hits"].sum()
    singles = df["singles"].sum()
    doubles = df["doubles"].sum()
    triples = df["triples"].sum()
    homers = df["home_runs"].sum()
    walks = df["walks"].sum()
    hbp = df["hbp"].sum()

    called_strikes = (df["called_strike_rate"] * df["pitches"]).fillna(0).sum()
    total_bases = singles + 2 * doubles + 3 * triples + 4 * homers

    whiff_rate = whiffs / swings if swings else np.nan
    called_strike_rate = called_strikes / pitches if pitches else np.nan
    strikeout_rate = strikeouts / pa if pa else np.nan
    k_rate_behind = _weighted_average(df["strikeout_rate_behind"], df["plate_appearances"])
    k_rate_ahead = _weighted_average(df["strikeout_rate_ahead"], df["plate_appearances"])
    avg = hits / ab if ab else np.nan
    obp = (hits + walks + hbp) / pa if pa else np.nan
    slugging = total_bases / ab if ab else np.nan
    ops = obp + slugging if not np.isnan(obp) and not np.isnan(slugging) else np.nan
    woba = _weighted_average(df["woba"], df["plate_appearances"])

    return {
        "bat_plate_appearances": pa,
        "bat_at_bats": ab,
        "bat_pitches": pitches,
        "bat_swings": swings,
        "bat_whiffs": whiffs,
        "bat_whiff_rate": whiff_rate,
        "bat_called_strike_rate": called_strike_rate,
        "bat_strikeouts": strikeouts,
        "bat_strikeout_rate": strikeout_rate,
        "bat_strikeout_rate_behind": k_rate_behind,
        "bat_strikeout_rate_ahead": k_rate_ahead,
        "bat_hits": hits,
        "bat_singles": singles,
        "bat_doubles": doubles,
        "bat_triples": triples,
        "bat_home_runs": homers,
        "bat_walks": walks,
        "bat_hbp": hbp,
        "bat_avg": avg,
        "bat_obp": obp,
        "bat_slugging": slugging,
        "bat_ops": ops,
        "bat_woba": woba,
    }

def build_team_batting_table(conn: sqlite3.Connection) -> pd.DataFrame:
    logger.info("Loading batter vs starter table %s", BATTERS_VS_STARTERS_TABLE)
    df = pd.read_sql_query(f"SELECT * FROM {BATTERS_VS_STARTERS_TABLE}", conn)
    if df.empty:
        logger.warning("No data found in %s", BATTERS_VS_STARTERS_TABLE)
        return pd.DataFrame()

    rows = []
    group_cols = ["game_pk", "pitching_team", "opponent_team"]
    for keys, grp in df.groupby(group_cols):
        feats = compute_team_batting_features(grp)
        feats.update({
            "game_pk": keys[0],
            "pitching_team": keys[1],
            "opponent_team": keys[2],
        })
        rows.append(feats)

    team_df = pd.DataFrame(rows)
    team_df.to_sql(TEAM_BATTING_TABLE, conn, if_exists="replace", index=False)
    logger.info("Wrote %d rows to %s", len(team_df), TEAM_BATTING_TABLE)
    return team_df

def build_matchup_table(conn: sqlite3.Connection, team_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Loading starting pitcher table %s", STARTERS_TABLE)
    pitch_df = pd.read_sql_query(f"SELECT * FROM {STARTERS_TABLE}", conn)
    if pitch_df.empty:
        logger.warning("No data found in %s", STARTERS_TABLE)
        return pd.DataFrame()

    merged = pitch_df.merge(
        team_df,
        on=["game_pk", "pitching_team", "opponent_team"],
        how="left",
    )
    merged.to_sql(MATCHUP_TABLE, conn, if_exists="replace", index=False)
    logger.info("Wrote %d rows to %s", len(merged), MATCHUP_TABLE)
    return merged

def main(db_path: Path = DBConfig.PATH) -> None:
    with DBConnection(db_path) as conn:
        team_df = build_team_batting_table(conn)
        if not team_df.empty:
            build_matchup_table(conn, team_df)

if __name__ == "__main__":
    main()
