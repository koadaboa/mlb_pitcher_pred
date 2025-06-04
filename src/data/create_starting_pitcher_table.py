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

# --- Pitch Type Groups ---
FASTBALL_TYPES = {"FF", "FA", "FT", "SI", "F4", "F2", "FC", "FS", "SF", "FO"}
BREAKING_TYPES = {"SL", "CU", "KC", "SV", "SC"}
OFFSPEED_TYPES = {"CH", "FO", "KN", "EP"}

logger = setup_logger(
    "create_starting_pitcher_table",
    LogConfig.LOG_DIR / "create_starting_pitcher_table.log",
)

# Log progress after this many games have been processed
LOG_EVERY_N = 100

# Number of worker processes for parallel aggregation
MAX_WORKERS = int(os.getenv("MAX_WORKERS", os.cpu_count() or 1))
# Chunk size for ``ProcessPoolExecutor.map``
CHUNK_SIZE = 100

# Select only required columns to reduce SQLite I/O
PITCHER_COLS = [
    "game_pk",
    "game_date",
    "pitcher",
    "batter",
    "p_throws",
    "home_team",
    "away_team",
    "inning_topbot",
    "at_bat_number",
    "pitch_number",
    "pitch_type",
    "description",
    "zone",
    "plate_x",
    "plate_z",
    "events",
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "release_extension",
    "launch_speed",
    "launch_angle",
    "inning",
    "type",
    "strikes",
    "balls",
    "home_score",
    "away_score",
    "on_1b",
    "on_2b",
    "on_3b",
    "woba_value",
    "woba_denom",
]


def filter_starting_pitchers(conn) -> pd.DataFrame:
    """Return game_pk/pitcher combos likely representing true starters."""
    query = """
        SELECT game_pk, pitcher
        FROM statcast_pitchers
        GROUP BY game_pk, pitcher
        HAVING MIN(inning) = 1
           AND COUNT(*) > 30
           AND MAX(inning) > 3
    """
    df = pd.read_sql_query(query, conn)
    logger.info("Found %d potential starting pitcher rows", len(df))
    return df


def load_pitcher_game(conn, game_pk: int, pitcher: int) -> pd.DataFrame:
    """Load all pitch-level rows for a pitcher in one game."""
    cols = ",".join(PITCHER_COLS)
    q = (
        f"SELECT {cols} FROM statcast_pitchers WHERE game_pk = ? AND pitcher = ?"
    )
    return pd.read_sql_query(q, conn, params=(game_pk, pitcher))


def compute_game_features(game_pk: int, pitcher: int, db_path: Path) -> Optional[Dict]:
    """Load one pitcher/game from SQLite and compute features."""
    with DBConnection(db_path) as conn:
        df = load_pitcher_game(conn, game_pk, pitcher)
    if df.empty:
        return None
    return compute_features(df)


def _map_compute_game_features(args: tuple[int, int, Path]) -> Optional[Dict]:
    """Wrapper for ``ProcessPoolExecutor.map``."""
    game_pk, pitcher, db_path = args
    return compute_game_features(game_pk, pitcher, db_path)


def compute_features(df: pd.DataFrame) -> Dict:
    df = df.sort_values(["at_bat_number", "pitch_number"])
    first_row = df.iloc[0]
    if first_row["inning_topbot"] == "Top":
        team = first_row["home_team"]
        opp = first_row["away_team"]
    else:
        team = first_row["away_team"]
        opp = first_row["home_team"]

    pitches = len(df)
    inning_counts = df.groupby("inning").size().sort_index()
    if len(inning_counts) > 1:
        slope = np.polyfit(inning_counts.index.values, inning_counts.values, 1)[0]
    else:
        slope = np.nan
    strike_events = df["events"].isin(["strikeout", "strikeout_double_play"])
    swinging = df["description"].str.contains("swinging_strike", na=False)
    called = df["description"].eq("called_strike")
    foul_tip = df["description"].eq("foul_tip")

    first_pitch = df[df["pitch_number"] == 1]
    first_pitch_strikes = first_pitch["type"].eq("S")

    fastball_mask = df["pitch_type"].isin(FASTBALL_TYPES)
    breaking_mask = df["pitch_type"].isin(BREAKING_TYPES)
    offspeed_mask = df["pitch_type"].isin(OFFSPEED_TYPES)

    # Use mutually exclusive pitch type sets for ratio calculations
    fastball_only = df["pitch_type"].isin(FASTBALL_TYPES - OFFSPEED_TYPES)
    offspeed_only = df["pitch_type"].isin(OFFSPEED_TYPES - FASTBALL_TYPES)

    # Pitch type category masks
    slider_mask = df["pitch_type"].eq("SL")
    curve_mask = df["pitch_type"].isin({"CU", "KC", "SV", "SC"})
    changeup_mask = df["pitch_type"].eq("CH")
    cutter_mask = df["pitch_type"].eq("FC")
    sinker_mask = df["pitch_type"].isin({"SI", "FT"})
    splitter_mask = df["pitch_type"].isin({"FS", "SF"})

    # Determine strike zone using Statcast zone or plate_x/plate_z
    if "zone" in df.columns:
        in_zone = df["zone"].between(1, 9)
    elif {"plate_x", "plate_z"}.issubset(df.columns):
        in_zone = df["plate_x"].between(-0.83, 0.83) & df["plate_z"].between(1.5, 3.5)
    else:
        in_zone = pd.Series([np.nan] * len(df))

    zone_pct = in_zone.mean() if not in_zone.isna().all() else np.nan

    swings_all = df["description"].str.contains("swing", case=False, na=False)
    chase_rate = (
        swings_all[~in_zone].mean()
        if (~in_zone).sum()
        else np.nan if not in_zone.isna().all() else np.nan
    )

    # Contact quality metrics on balls in play
    in_play_mask = df["type"].eq("X") if "type" in df.columns else pd.Series(False)
    bip = df[in_play_mask]
    avg_launch_speed = (
        bip["launch_speed"].mean() if "launch_speed" in bip.columns else np.nan
    )
    max_launch_speed = (
        bip["launch_speed"].max() if "launch_speed" in bip.columns else np.nan
    )
    avg_launch_angle = (
        bip["launch_angle"].mean() if "launch_angle" in bip.columns else np.nan
    )
    max_launch_angle = (
        bip["launch_angle"].max() if "launch_angle" in bip.columns else np.nan
    )
    hard_hit_rate = (
        (bip["launch_speed"] >= 95).mean()
        if "launch_speed" in bip.columns and len(bip)
        else np.nan
    )
    if "barrel" in bip.columns and len(bip):
        barrel_rate = bip["barrel"].mean()
    elif {"launch_speed", "launch_angle"}.issubset(bip.columns) and len(bip):
        is_barrel = (
            (bip["launch_angle"].between(26, 30))
            & (bip["launch_speed"] >= 98)
        )
        barrel_rate = is_barrel.mean()
    else:
        barrel_rate = np.nan

    types = df["pitch_type"].values
    next_types = np.roll(types, -1)
    fastball_then_break = fastball_mask & np.isin(next_types, list(BREAKING_TYPES))

    features = {
        "game_pk": first_row["game_pk"],
        "game_date": first_row["game_date"],
        "pitcher_id": first_row["pitcher"],
        "pitcher_hand": first_row["p_throws"],
        "pitching_team": team,
        "opponent_team": opp,
        "pitches": pitches,
        "pitches_per_inning_decay": slope,
        "innings_pitched": df["inning"].nunique(),
        "batters_faced": df["batter"].nunique(),
        "strikeouts": strike_events.sum(),
        "swinging_strike_rate": swinging.mean(),
        "first_pitch_strike_rate": (
            first_pitch_strikes.mean() if len(first_pitch) else np.nan
        ),
        "csw_pct": ((called | swinging | foul_tip).mean()),
        "fastball_pct": fastball_only.mean(),
        "offspeed_to_fastball_ratio": (
            offspeed_only.sum() / fastball_only.sum() if fastball_only.sum() else np.nan
        ),
        "fastball_then_breaking_rate": (
            fastball_then_break[:-1].mean() if len(df) > 1 else np.nan
        ),
        "avg_release_speed": df["release_speed"].mean(),
        "max_release_speed": df["release_speed"].max(),
        "avg_spin_rate": df["release_spin_rate"].mean(),
        "unique_pitch_types": df["pitch_type"].nunique(),
        "zone_pct": zone_pct,
        "chase_rate": chase_rate,
        "avg_launch_speed": avg_launch_speed,
        "max_launch_speed": max_launch_speed,
        "avg_launch_angle": avg_launch_angle,
        "max_launch_angle": max_launch_angle,
        "hard_hit_rate": hard_hit_rate,
        "barrel_rate": barrel_rate,
        "pfx_x": df["pfx_x"].mean() if "pfx_x" in df.columns else np.nan,
        "pfx_z": df["pfx_z"].mean() if "pfx_z" in df.columns else np.nan,
        "release_extension": df["release_extension"].mean()
        if "release_extension" in df.columns
        else np.nan,
        "plate_x": df["plate_x"].mean() if "plate_x" in df.columns else np.nan,
        "plate_z": df["plate_z"].mean() if "plate_z" in df.columns else np.nan,
        # FIP formula without constant: (13*HR + 3*(BB+HBP) - 2*K) / IP
        "fip": (
            (
                13 * df["events"].eq("home_run").sum()
                + 3
                * (
                    df["events"].isin(["walk", "intent_walk"]).sum()
                    + df["events"].eq("hit_by_pitch").sum()
                )
                - 2 * strike_events.sum()
            )
            / df["inning"].nunique()
            if df["inning"].nunique()
            else np.nan
        ),
    }

    # --- Pitch Usage Percentages ---
    for name, mask in {
        "slider": slider_mask,
        "curve": curve_mask,
        "changeup": changeup_mask,
        "cutter": cutter_mask,
        "sinker": sinker_mask,
        "splitter": splitter_mask,
    }.items():
        features[f"{name}_pct"] = mask.mean()
        features[f"{name}_whiff_rate"] = swinging[mask].mean() if mask.sum() else np.nan

    # Fastball whiff rate
    features["fastball_whiff_rate"] = (
        swinging[fastball_only].mean() if fastball_only.sum() else np.nan
    )

    last_pitch = df.groupby("at_bat_number").tail(1)
    if "strikes" in last_pitch.columns:
        mask = last_pitch["strikes"] == 2
        if mask.any():
            features["two_strike_k_rate"] = (
                last_pitch.loc[mask, "events"].isin(["strikeout", "strikeout_double_play"]).mean()
            )
        else:
            features["two_strike_k_rate"] = np.nan
    else:
        features["two_strike_k_rate"] = np.nan

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
            if first_row["inning_topbot"] == "Top"
            else last_pitch["away_score"] - last_pitch["home_score"]
        )
        lev_mask = (last_pitch["inning"] >= 7) & (run_diff.abs() <= 1)
        if lev_mask.any():
            high_lev_rate = last_pitch.loc[lev_mask, "events"].isin([
                "strikeout",
                "strikeout_double_play",
            ]).mean()
    features["high_leverage_k_rate"] = high_lev_rate

    if {
        "on_1b",
        "on_2b",
        "on_3b",
        "woba_value",
        "woba_denom",
    }.issubset(df.columns):
        first_ab = df.sort_values("pitch_number").groupby("at_bat_number").first()
        on_base = first_ab[["on_1b", "on_2b", "on_3b"]].notna().any(axis=1)
        woba_val = df.groupby("at_bat_number")["woba_value"].sum()
        woba_den = df.groupby("at_bat_number")["woba_denom"].sum()
        denom = woba_den[on_base].sum()
        features["woba_runners_on"] = (
            woba_val[on_base].sum() / denom if denom else np.nan
        )
    else:
        features["woba_runners_on"] = np.nan
    return features


def aggregate_to_game_level(db_path: Path = DBConfig.PATH) -> pd.DataFrame:
    with DBConnection(db_path) as conn:
        starters = filter_starting_pitchers(conn)
        total_games = len(starters)

    result_rows: list[Dict] = []
    pairs = [tuple(row) + (db_path,) for row in starters.itertuples(index=False, name=None)]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exc:
        for processed, res in enumerate(
            exc.map(_map_compute_game_features, pairs, chunksize=CHUNK_SIZE),
            1,
        ):
            if res:
                result_rows.append(res)
            if processed % LOG_EVERY_N == 0:
                logger.info("Processed %d/%d games", processed, total_games)

    game_df = pd.DataFrame(result_rows)
    with DBConnection(db_path) as conn:
        game_df.to_sql(
            "game_level_starting_pitchers",
            conn,
            index=False,
            if_exists="replace",
        )
    return game_df


def main() -> None:
    try:
        df = aggregate_to_game_level()
        logger.info(f"Aggregated {len(df)} starting pitcher games")
    except Exception as exc:
        logger.exception("Failed to create starting pitcher table: %s", exc)


if __name__ == "__main__":
    main()
