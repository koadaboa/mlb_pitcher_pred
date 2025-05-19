import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import DBConnection

try:
    from src.config import DBConfig
except Exception:
    class DBConfig:
        PATH = "data/pitcher_stats.db"

FASTBALLS = {
    "FF", "FA", "FT", "SI", "FC", "FS", "FO"
}
BREAKING_BALLS = {
    "SL", "CU", "KC", "KN", "CS", "ST"
}

STARTERS_TABLE = "game_level_starting_pitchers"
PITCHERS_TABLE = "statcast_pitchers"

def get_starting_pitchers(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return DataFrame of starting pitcher IDs per game and team."""
    query = f"""
    WITH pitch_team AS (
        SELECT ROWID AS rid,
               game_pk,
               CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END AS pitching_team,
               pitcher_id
        FROM {PITCHERS_TABLE}
    ), first_pitch AS (
        SELECT game_pk, pitching_team, MIN(rid) AS min_rid
        FROM pitch_team
        GROUP BY game_pk, pitching_team
    )
    SELECT pt.game_pk, pt.pitching_team, pt.pitcher_id
    FROM pitch_team pt
    JOIN first_pitch fp
      ON pt.game_pk = fp.game_pk
     AND pt.pitching_team = fp.pitching_team
     AND pt.rid = fp.min_rid
    """
    return pd.read_sql_query(query, conn)

def compute_features(df: pd.DataFrame) -> dict:
    df = df.sort_values(["at_bat_number", "pitch_number"]).reset_index(drop=True)
    num_pitches = len(df)
    if num_pitches == 0:
        return {}
    max_inning = df["inning"].max()
    max_outs = df.loc[df["inning"] == max_inning, "outs_when_up"].max()
    innings_pitched = (max_inning - 1) + (max_outs or 0) / 3.0
    strikeouts = df["events"].str.contains("strikeout", case=False, na=False).sum()
    swinging = df["description"].str.contains("swinging_strike", case=False, na=False).sum()
    swinging_rate = swinging / num_pitches
    # first pitch strike rate
    first_pitch = df[df["pitch_number"] == 1]
    if not first_pitch.empty:
        fp_strike = first_pitch["description"].str.contains("strike", case=False, na=False).mean()
    else:
        fp_strike = np.nan
    fastball_pct = df["pitch_type"].isin(FASTBALLS).mean()
    # fastball then breaking ball transitions
    df["prev_pitch"] = df["pitch_type"].shift(1)
    seq_trans = ((df["prev_pitch"].isin(FASTBALLS)) & (df["pitch_type"].isin(BREAKING_BALLS))).sum()
    seq_rate = seq_trans / max(num_pitches - 1, 1)
    return {
        "innings_pitched": innings_pitched,
        "pitches": num_pitches,
        "strikeouts": strikeouts,
        "swinging_strike_rate": swinging_rate,
        "first_pitch_strike_rate": fp_strike,
        "fastball_pct": fastball_pct,
        "fastball_then_breaking_rate": seq_rate,
    }

def main(db_path: Path = DBConfig.PATH) -> None:
    """Build or replace the aggregated starting pitcher table."""
    with DBConnection(db_path) as conn:
        starters = get_starting_pitchers(conn)
        rows = []
        for _, s in starters.iterrows():
            df = pd.read_sql_query(
                f"SELECT pitch_type, at_bat_number, pitch_number, events, description, inning, inning_topbot, outs_when_up FROM {PITCHERS_TABLE} WHERE game_pk=? AND pitcher_id=?",
                conn,
                params=(s.game_pk, s.pitcher_id),
            )
            feats = compute_features(df)
            if not feats:
                continue
            if feats["innings_pitched"] < 3 and feats["pitches"] < 50:
                continue
            feats.update({
                "game_pk": s.game_pk,
                "pitcher_id": s.pitcher_id,
                "pitching_team": s.pitching_team,
            })
            rows.append(feats)
        if rows:
            out_df = pd.DataFrame(rows)
            out_df.to_sql(STARTERS_TABLE, conn, if_exists="replace", index=False)

if __name__ == "__main__":
    main()
