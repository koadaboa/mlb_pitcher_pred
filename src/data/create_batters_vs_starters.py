from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional

from src.utils import DBConnection, setup_logger
from src.config import DBConfig, LogConfig

logger = setup_logger(
    "create_batters_vs_starters",
    LogConfig.LOG_DIR / "create_batters_vs_starters.log",
)

# Progress logging
LOG_EVERY_N = 100
MAX_WORKERS = 9
BATCH_SIZE = 5000

# --- Helper functions ---

def filter_starting_pitchers(conn) -> pd.DataFrame:
    """Return game/pitcher combos likely representing true starters."""
    query = """
        SELECT game_pk, pitcher
        FROM statcast_pitchers
        GROUP BY game_pk, pitcher
        HAVING MIN(inning) = 1
           AND COUNT(*) > 30
           AND MAX(inning) > 3
    """
    df = pd.read_sql_query(query, conn)
    logger.info("Found %d potential starters", len(df))
    return df


def load_batter_game(conn, game_pk: int, pitcher: int) -> pd.DataFrame:
    """Load all batter rows for a pitcher in one game."""
    q = "SELECT * FROM statcast_batters WHERE game_pk = ? AND pitcher = ?"
    return pd.read_sql_query(q, conn, params=(game_pk, pitcher))


def compute_batter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one game's batter data against the starter and return a DataFrame."""
    """Aggregate one game's batter data against the starter."""
    df = df.sort_values(["batter", "at_bat_number", "pitch_number"])
    first = df.iloc[0]
    if first["inning_topbot"] == "Top":
        pitching_team = first["home_team"]
        opponent_team = first["away_team"]
    else:
        pitching_team = first["away_team"]
        opponent_team = first["home_team"]

    records: list[Dict] = []
    for batter_id, bdf in df.groupby("batter"):
        last_pitch = bdf.sort_values("pitch_number").groupby("at_bat_number").tail(1)
        plate_appearances = len(last_pitch)

        official_ab_mask = ~last_pitch["events"].isin(
            [
                "walk",
                "intent_walk",
                "hit_by_pitch",
                "catcher_interf",
                "sac_fly",
                "sac_bunt",
            ]
        )
        at_bats = official_ab_mask.sum()

        events = last_pitch["events"]
        hits = events.isin(["single", "double", "triple", "home_run"]).sum()
        singles = events.eq("single").sum()
        doubles = events.eq("double").sum()
        triples = events.eq("triple").sum()
        home_runs = events.eq("home_run").sum()
        walks = events.isin(["walk", "intent_walk"]).sum()
        hbp = events.eq("hit_by_pitch").sum()
        strikeouts = events.isin(["strikeout", "strikeout_double_play"]).sum()

        behind_mask = last_pitch["strikes"] > last_pitch["balls"]
        ahead_mask = last_pitch["balls"] > last_pitch["strikes"]
        strikeout_rate_behind = (
            strikeouts if behind_mask.sum() else np.nan
        )
        if behind_mask.sum():
            strikeout_rate_behind = (
                last_pitch[behind_mask]["events"].isin(["strikeout", "strikeout_double_play"]).mean()
            )
        strikeout_rate_ahead = (
            strikeouts if ahead_mask.sum() else np.nan
        )
        if ahead_mask.sum():
            strikeout_rate_ahead = (
                last_pitch[ahead_mask]["events"].isin(["strikeout", "strikeout_double_play"]).mean()
            )

        pitches = len(bdf)
        swings = bdf["description"].str.contains("swing", case=False, na=False)
        whiffs = bdf["description"].str.contains("swinging_strike", case=False, na=False)
        swings_total = swings.sum()
        whiffs_total = whiffs.sum()

        row = {
            "game_pk": first["game_pk"],
            "batter_id": batter_id,
            "pitcher_id": first["pitcher"],
            "pitching_team": pitching_team,
            "opponent_team": opponent_team,
            "plate_appearances": plate_appearances,
            "at_bats": at_bats,
            "pitches": pitches,
            "swings": swings_total,
            "whiffs": whiffs_total,
            "whiff_rate": whiffs_total / swings_total if swings_total else np.nan,
            "called_strike_rate": bdf["description"].eq("called_strike").mean(),
            "strikeouts": strikeouts,
            "strikeout_rate": strikeouts / plate_appearances if plate_appearances else np.nan,
            "strikeout_rate_behind": strikeout_rate_behind,
            "strikeout_rate_ahead": strikeout_rate_ahead,
            "hits": hits,
            "singles": singles,
            "doubles": doubles,
            "triples": triples,
            "home_runs": home_runs,
            "walks": walks,
            "hbp": hbp,
        }

        # --- Rate stats ---
        row["avg"] = hits / at_bats if at_bats else np.nan
        row["obp"] = (hits + walks + hbp) / plate_appearances if plate_appearances else np.nan
        row["slugging"] = (
            singles + 2 * doubles + 3 * triples + 4 * home_runs
        ) / at_bats if at_bats else np.nan
        row["ops"] = row["obp"] + row["slugging"] if plate_appearances and at_bats else np.nan

        if "woba_value" in bdf.columns and "woba_denom" in bdf.columns:
            woba_denom = bdf["woba_denom"].sum()
            row["woba"] = bdf["woba_value"].sum() / woba_denom if woba_denom else np.nan
        else:
            row["woba"] = np.nan

        records.append(row)
    return pd.DataFrame.from_records(records)


def compute_game_features(game_pk: int, pitcher: int, db_path: Path) -> Optional[pd.DataFrame]:
    with DBConnection(db_path) as conn:
        df = load_batter_game(conn, game_pk, pitcher)
    if df.empty:
        return None
    return compute_batter_rows(df)


def aggregate_to_game_level(db_path: Path = DBConfig.PATH) -> pd.DataFrame:
    with DBConnection(db_path) as conn:
        starters = filter_starting_pitchers(conn)
        total_games = len(starters)

    frames: list[pd.DataFrame] = []
    rows: list[Dict] = []
    processed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exc:
        pending = []
        for game_pk, pitcher in starters.itertuples(index=False):
            pending.append(exc.submit(compute_game_features, game_pk, pitcher, db_path))
            if len(pending) >= BATCH_SIZE:
                for fut in as_completed(pending):
                    res = fut.result()
                    processed += 1
                    if res is not None:
                        frames.append(res)
                    if processed % LOG_EVERY_N == 0:
                        logger.info("Processed %d/%d games", processed, total_games)
                pending.clear()

        for fut in as_completed(pending):
            res = fut.result()
            processed += 1
            if res is not None:
                frames.append(res)
            if processed % LOG_EVERY_N == 0:
                logger.info("Processed %d/%d games", processed, total_games)

    if frames:
        game_df = pd.concat(frames, ignore_index=True)
    else:
        game_df = pd.DataFrame()
            if res:
                rows.extend(res)
            if processed % LOG_EVERY_N == 0:
                logger.info("Processed %d/%d games", processed, total_games)

    game_df = pd.DataFrame(rows)
    with DBConnection(db_path) as conn:
        game_df.to_sql(
            "game_level_batters_vs_starters",
            conn,
            index=False,
            if_exists="replace",
        )
    return game_df


def main() -> None:
    try:
        df = aggregate_to_game_level()
        logger.info("Aggregated %d batter vs starter rows", len(df))
    except Exception as exc:
        logger.exception("Failed to create batter vs starter table: %s", exc)


if __name__ == "__main__":
    main()
