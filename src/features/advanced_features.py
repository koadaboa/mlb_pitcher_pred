from __future__ import annotations

import numpy as np
import pandas as pd

TEAM_DIVISIONS = {
    "ARI": "NL West",
    "ATL": "NL East",
    "BAL": "AL East",
    "BOS": "AL East",
    "CHC": "NL Central",
    "CWS": "AL Central",
    "CIN": "NL Central",
    "CLE": "AL Central",
    "COL": "NL West",
    "DET": "AL Central",
    "HOU": "AL West",
    "KC": "AL Central",
    "LAA": "AL West",
    "LAD": "NL West",
    "MIA": "NL East",
    "MIL": "NL Central",
    "MIN": "AL Central",
    "NYM": "NL East",
    "NYY": "AL East",
    "OAK": "AL West",
    "PHI": "NL East",
    "PIT": "NL Central",
    "SD": "NL West",
    "SF": "NL West",
    "SEA": "AL West",
    "STL": "NL Central",
    "TB": "AL East",
    "TEX": "AL West",
    "TOR": "AL East",
    "WSH": "NL East",
}


def is_division_rival(team_a: str | None, team_b: str | None) -> bool:
    """Return True if ``team_a`` and ``team_b`` belong to the same division."""
    if not team_a or not team_b:
        return False
    return TEAM_DIVISIONS.get(team_a) == TEAM_DIVISIONS.get(team_b)


def calculate_days_since_all_star_break(dates: pd.Series) -> pd.Series:
    """Return days since the season's All-Star break (approx mid-July)."""
    year = dates.dt.year
    asb = pd.to_datetime(year.astype(str) + "-07-15")
    out = (dates - asb).dt.days
    return out.clip(lower=0)


def calculate_linear_trend(series: pd.Series) -> float:
    """Return slope of a linear trend for ``series``."""
    series = series.dropna()
    if len(series) < 3:
        return 0.0
    x = np.arange(len(series))
    return float(np.polyfit(x, series, 1)[0])


def calculate_acceleration(series: pd.Series) -> float:
    """Return average second derivative of ``series``."""
    series = series.dropna()
    if len(series) < 3:
        return 0.0
    first_diff = np.diff(series)
    if len(first_diff) < 2:
        return 0.0
    second_diff = np.diff(first_diff)
    return float(np.mean(second_diff))


def detect_change_points(series: pd.Series, threshold: float = 1.5) -> int:
    """Detect regime changes using a simple CUSUM approach."""
    series = series.dropna()
    if len(series) < 10:
        return 0
    mean_val = series.mean()
    cusum_pos = 0.0
    cusum_neg = 0.0
    changes = 0
    for val in series:
        cusum_pos = max(0.0, cusum_pos + (val - mean_val))
        cusum_neg = min(0.0, cusum_neg + (val - mean_val))
        if cusum_pos > threshold or cusum_neg < -threshold:
            changes += 1
            cusum_pos = 0.0
            cusum_neg = 0.0
    return changes


def add_advanced_rolling_features(
    df: pd.DataFrame,
    group_col: str = "pitcher_id",
    date_col: str = "game_date",
) -> pd.DataFrame:
    """Add advanced rolling window statistics to ``df``."""
    df = df.sort_values([group_col, date_col])
    grouped = df.groupby(group_col)
    frames = [df]

    # Trend features
    for window in [5, 10, 20]:
        for col in ["strikeouts", "swinging_strike_rate", "first_pitch_strike_rate"]:
            if col not in df.columns:
                continue
            trend = grouped[col].apply(
                lambda x: x.shift(1).rolling(window).apply(calculate_linear_trend)
            )
            frames.append(trend.rename(f"{col}_trend_{window}"))
    if "strikeouts" in df.columns:
        accel = grouped["strikeouts"].apply(
            lambda x: x.shift(1).rolling(10).apply(calculate_acceleration)
        )
        frames.append(accel.rename("strikeouts_acceleration_10"))

    # Volatility features
    for window in [5, 10]:
        for col in ["strikeouts", "swinging_strike_rate"]:
            if col not in df.columns:
                continue
            roll = grouped[col].shift(1).rolling(window)
            cv = roll.std() / roll.mean()
            frames.append(cv.rename(f"{col}_cv_{window}"))
    if "strikeouts" in df.columns:
        for window in [10, 20]:
            roll = grouped["strikeouts"].shift(1).rolling(window)
            rng = roll.max() - roll.min()
            frames.append(rng.rename(f"strikeouts_range_{window}"))

    # Regime change
    if "strikeouts" in df.columns:
        regime = grouped["strikeouts"].apply(lambda x: detect_change_points(x.shift(1)))
        frames.append(regime.rename("performance_regime_change"))
        recent = grouped["strikeouts"].shift(1).rolling(5).mean()
        long = grouped["strikeouts"].shift(1).rolling(50).mean()
        frames.append((recent - long).rename("recent_vs_longterm_divergence"))

    # Seasonal features
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        month = df[date_col].dt.month
        frames.append(((month >= 8).astype(int)).rename("late_season_fatigue"))
        frames.append(calculate_days_since_all_star_break(df[date_col]).rename("days_since_asb"))

    if {"pitching_team", "opponent_team"}.issubset(df.columns):
        div = df.apply(
            lambda x: int(is_division_rival(x["pitching_team"], x["opponent_team"])),
            axis=1,
        )
        frames.append(div.rename("division_rival"))

    result = pd.concat(frames, axis=1)
    return result


def add_game_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add in-game context features such as fatigue and score effects."""
    if "pitches" in df.columns:
        df["pitch_count_fatigue"] = np.where(df["pitches"] > 100, (df["pitches"] - 100) / 20, 0)
    if {"home_score", "away_score"}.issubset(df.columns):
        df["score_diff_abs"] = (df["home_score"] - df["away_score"]).abs()
        df["close_game"] = (df["score_diff_abs"] <= 2).astype(int)
    if "n_thruorder_pitcher" in df.columns:
        df["times_through_order_penalty"] = (
            df.groupby("pitcher_id")["n_thruorder_pitcher"].apply(
                lambda x: x.shift(1).rolling(10).apply(lambda y: (y >= 3).mean())
            )
        )
    return df
