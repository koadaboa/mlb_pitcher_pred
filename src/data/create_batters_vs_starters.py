from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import logging

from typing import Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from src.utils import DBConnection, setup_logger
from src.config import DBConfig, LogConfig

logger = setup_logger(
    "create_batters_vs_starters",
    LogConfig.LOG_DIR / "create_batters_vs_starters.log",
)

# Progress logging
LOG_EVERY_N = 100
MAX_WORKERS = int(os.getenv("MAX_WORKERS", os.cpu_count() or 1))
CHUNK_SIZE = 100

# Only fetch required columns
BATTER_COLS = [
    "game_pk",
    "game_date",
    "batter",
    "pitcher",
    "stand",
    "inning_topbot",
    "home_team",
    "away_team",
    "at_bat_number",
    "pitch_number",
    "description",
    "events",
    "balls",
    "strikes",
    "leverage_index",
    "inning",
    "home_score",
    "away_score",
    "on_1b",
    "on_2b",
    "on_3b",
    "woba_value",
    "woba_denom",
]

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
    pragma = pd.read_sql_query("PRAGMA table_info(statcast_batters)", conn)
    available = set(pragma["name"].tolist())
    cols = [c for c in BATTER_COLS if c in available]
    q = f"SELECT {','.join(cols)} FROM statcast_batters WHERE game_pk = ? AND pitcher = ?"
    return pd.read_sql_query(q, conn, params=(game_pk, pitcher))


def compute_batter_rows(df: pd.DataFrame) -> list[Dict]:
    """Aggregate one game's batter data against the starter."""
    df = df.sort_values(["batter", "at_bat_number", "pitch_number"])
    first = df.iloc[0]
    if first["inning_topbot"] == "Top":
        pitching_team = first["home_team"]
        opponent_team = first["away_team"]
    else:
        pitching_team = first["away_team"]
        opponent_team = first["home_team"]
    rows = []

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
        strikeout_rate_behind = strikeouts if behind_mask.sum() else np.nan
        if behind_mask.sum():
            strikeout_rate_behind = (
                last_pitch[behind_mask]["events"]
                .isin(["strikeout", "strikeout_double_play"])
                .mean()
            )
        strikeout_rate_ahead = strikeouts if ahead_mask.sum() else np.nan
        if ahead_mask.sum():
            strikeout_rate_ahead = (
                last_pitch[ahead_mask]["events"]
                .isin(["strikeout", "strikeout_double_play"])
                .mean()
            )

        pitches = len(bdf)
        swings = bdf["description"].str.contains("swing", case=False, na=False)
        whiffs = bdf["description"].str.contains(
            "swinging_strike", case=False, na=False
        )
        swings_total = swings.sum()
        whiffs_total = whiffs.sum()

        two_strike_rate = np.nan
        if "strikes" in last_pitch.columns:
            mask_two = last_pitch["strikes"] == 2
            if mask_two.any():
                two_strike_rate = last_pitch.loc[mask_two, "events"].isin([
                    "strikeout",
                    "strikeout_double_play",
                ]).mean()

        high_lev_rate = np.nan
        if "leverage_index" in last_pitch.columns:
            lev_mask = last_pitch["leverage_index"] >= 1.5
            if lev_mask.any():
                high_lev_rate = last_pitch.loc[lev_mask, "events"].isin([
                    "strikeout",
                    "strikeout_double_play",
                ]).mean()
        elif {"home_score", "away_score", "inning"}.issubset(last_pitch.columns):
            run_diff = (
                last_pitch["home_score"] - last_pitch["away_score"]
                if first["inning_topbot"] == "Top"
                else last_pitch["away_score"] - last_pitch["home_score"]
            )
            lev_mask = (last_pitch["inning"] >= 7) & (run_diff.abs() <= 1)
            if lev_mask.any():
                high_lev_rate = last_pitch.loc[lev_mask, "events"].isin([
                    "strikeout",
                    "strikeout_double_play",
                ]).mean()

        woba_runners = np.nan
        if {
            "on_1b",
            "on_2b",
            "on_3b",
            "woba_value",
            "woba_denom",
        }.issubset(bdf.columns):
            first_ab = (
                bdf.sort_values("pitch_number").groupby("at_bat_number").first()
            )
            on_base = first_ab[["on_1b", "on_2b", "on_3b"]].notna().any(axis=1)
            woba_val = bdf.groupby("at_bat_number")["woba_value"].sum()
            woba_den = bdf.groupby("at_bat_number")["woba_denom"].sum()
            denom = woba_den[on_base].sum()
            woba_runners = (
                woba_val[on_base].sum() / denom if denom else np.nan
            )

        row = {
            "game_pk": first["game_pk"],
            "batter_id": batter_id,
            "pitcher_id": first["pitcher"],
            "stand": bdf.iloc[0]["stand"],
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
            "strikeout_rate": (
                strikeouts / plate_appearances if plate_appearances else np.nan
            ),
            "strikeout_rate_behind": strikeout_rate_behind,
            "strikeout_rate_ahead": strikeout_rate_ahead,
            "hits": hits,
            "singles": singles,
            "doubles": doubles,
            "triples": triples,
            "home_runs": home_runs,
            "walks": walks,
            "hbp": hbp,
            "two_strike_k_rate": two_strike_rate,
            "high_leverage_k_rate": high_lev_rate,
            "woba_runners_on": woba_runners,
        }

        # --- Rate stats ---
        row["avg"] = hits / at_bats if at_bats else np.nan
        row["obp"] = (
            (hits + walks + hbp) / plate_appearances if plate_appearances else np.nan
        )
        row["slugging"] = (
            (singles + 2 * doubles + 3 * triples + 4 * home_runs) / at_bats
            if at_bats
            else np.nan
        )
        row["ops"] = (
            row["obp"] + row["slugging"] if plate_appearances and at_bats else np.nan
        )

        if "woba_value" in bdf.columns and "woba_denom" in bdf.columns:
            woba_denom = bdf["woba_denom"].sum()
            row["woba"] = bdf["woba_value"].sum() / woba_denom if woba_denom else np.nan
        else:
            row["woba"] = np.nan
        rows.append(row)
    return rows


def compute_game_features(
    game_pk: int, pitcher: int, db_path: Path
) -> Optional[list[Dict]]:
    with DBConnection(db_path) as conn:
        df = load_batter_game(conn, game_pk, pitcher)
    if df.empty:
        return None
    return compute_batter_rows(df)


def _map_compute_game_features(args: tuple[int, int, Path]) -> Optional[list[Dict]]:
    """Wrapper for ``ProcessPoolExecutor.map``."""
    game_pk, pitcher, db_path = args
    return compute_game_features(game_pk, pitcher, db_path)


def aggregate_to_game_level(db_path: Path = DBConfig.PATH) -> pd.DataFrame:
    with DBConnection(db_path) as conn:
        starters = filter_starting_pitchers(conn)
        total_games = len(starters)

    rows: list[Dict] = []
    pairs = [tuple(row) + (db_path,) for row in starters.itertuples(index=False, name=None)]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exc:
        for processed, res in enumerate(
            exc.map(_map_compute_game_features, pairs, chunksize=CHUNK_SIZE),
            1,
        ):
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
