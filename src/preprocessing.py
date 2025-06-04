import pandas as pd
import numpy as np


def improve_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Perform outlier handling, missing value imputation, and normalization."""

    def handle_outliers_intelligently(df: pd.DataFrame) -> pd.DataFrame:
        """Context-aware outlier handling."""
        df = df.copy()
        if "strikeouts" in df.columns:
            df["strikeouts_capped"] = df.groupby("pitcher_id")["strikeouts"].transform(
                lambda x: np.clip(x, x.quantile(0.01), x.quantile(0.99))
            )
            if "batters_faced" in df.columns:
                df = df[df["strikeouts"] <= df["batters_faced"]]
        if "pitches" in df.columns:
            df["valid_game"] = (df["pitches"] >= 20) & (df["pitches"] <= 140)
        if "innings_pitched" in df.columns:
            df = df[df["innings_pitched"] > 0]
        return df

    def handle_missing_values_better(df: pd.DataFrame) -> pd.DataFrame:
        """Context-aware missing value imputation."""
        df = df.copy()
        pitcher_specific_cols = [
            "swinging_strike_rate",
            "first_pitch_strike_rate",
            "slider_pct",
            "fastball_pct",
        ]
        for col in pitcher_specific_cols:
            if col in df.columns:
                df[col] = df.groupby("pitcher_id")[col].transform(lambda x: x.fillna(x.mean()))
                df[col] = df[col].fillna(df[col].mean())
        opponent_cols = ["bat_avg", "bat_ops", "bat_whiff_rate"]
        for col in opponent_cols:
            if col in df.columns:
                df[col] = df.groupby("opponent_team")[col].transform(lambda x: x.fillna(x.mean()))
                df[col] = df[col].fillna(df[col].mean())
        weather_cols = ["temp", "wind_speed", "humidity"]
        if "game_date" in df.columns:
            month = df["game_date"].dt.month
        else:
            month = None
        for col in weather_cols:
            if col in df.columns:
                if month is not None:
                    df[col] = df.groupby(["home_team", month])[col].transform(lambda x: x.fillna(x.mean()))
                df[col] = df[col].fillna(df[col].mean())
        return df

    def normalize_features_contextually(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features within meaningful contexts."""
        df = df.copy()
        pitcher_features = [
            "swinging_strike_rate",
            "first_pitch_strike_rate",
            "whiff_rate",
        ]
        for col in pitcher_features:
            if col in df.columns:
                df[f"{col}_zscore_self"] = df.groupby("pitcher_id")[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                )
                df[f"{col}_percentile"] = df[col].rank(pct=True)
        for col in ["bat_ops", "bat_avg", "bat_whiff_rate"]:
            if col in df.columns:
                df[f"{col}_zscore_league"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
        return df

    df = handle_outliers_intelligently(df)
    df = handle_missing_values_better(df)
    df = normalize_features_contextually(df)
    return df


def add_data_validation_checks(df: pd.DataFrame) -> list[str]:
    """Return list of issues detected in ``df``."""
    issues: list[str] = []
    if "strikeouts" in df.columns and "batters_faced" in df.columns:
        if (df["strikeouts"] > df["batters_faced"]).any():
            issues.append("Strikeouts > batters faced detected")
    if "innings_pitched" in df.columns:
        if (df["innings_pitched"] < 0).any():
            issues.append("Negative innings pitched detected")
    if "strikeouts" in df.columns and df["strikeouts"].std() == 0:
        issues.append("No variance in strikeouts - check data")
    critical_features = ["pitcher_id", "game_date", "strikeouts", "opponent_team"]
    for feat in critical_features:
        if feat not in df.columns:
            issues.append(f"Missing critical feature: {feat}")
        elif df[feat].isna().sum() > len(df) * 0.1:
            issues.append(f"High missing rate for {feat}: {df[feat].isna().mean():.2%}")
    if "game_date" in df.columns and df["game_date"].isna().any():
        issues.append("Missing game dates detected")
    if issues:
        print("Data Quality Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    return issues


def add_feature_engineering_from_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract additional signal from existing raw data."""
    df = df.copy()
    if "pitch_type" in df.columns:
        df["pitch_type_entropy"] = df.groupby(["pitcher_id", "game_pk"])["pitch_type"].transform(
            lambda x: calculate_entropy(x.value_counts())
        )
        if "release_pos_x" in df.columns and "release_pos_z" in df.columns:
            df["pitch_tunneling_score"] = (
                df.groupby(["pitcher_id", "game_pk"]).apply(calculate_pitch_tunneling_score).reindex(df.index)
            )
    df = _add_situational_context(df)
    df = _add_advanced_park_factors(df)
    return df


def calculate_entropy(counts: pd.Series) -> float:
    """Return Shannon entropy of ``counts``."""
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities + 1e-8)).sum())


def calculate_pitch_tunneling_score(group: pd.DataFrame) -> float:
    """Return pitch tunneling effectiveness for a group."""
    if "release_pos_x" not in group.columns or "release_pos_z" not in group.columns:
        return 0.0
    x_var = group["release_pos_x"].var()
    z_var = group["release_pos_z"].var()
    return float(1 / (1 + x_var + z_var))


def calculate_game_importance(df: pd.DataFrame) -> pd.Series:
    """Estimate relative game importance."""
    importance = np.ones(len(df))
    if "game_date" in df.columns:
        importance += (df["game_date"].dt.dayofyear > 244) * 0.2
    return importance


def _add_situational_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["game_importance"] = calculate_game_importance(df)
    if "game_date" in df.columns:
        df["days_rest"] = df.groupby("pitcher_id")["game_date"].diff().dt.days
    if "pitches" in df.columns:
        df["recent_workload"] = df.groupby("pitcher_id")["pitches"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    df["pitcher_vs_team_games"] = df.groupby(["pitcher_id", "opponent_team"]).cumcount()
    return df


def _add_advanced_park_factors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "elevation" in df.columns and "breaking_ball_pct" in df.columns:
        df["altitude_breaking_effect"] = df["elevation"] * df["breaking_ball_pct"] / 1000
    if "humidity" in df.columns and "breaking_ball_pct" in df.columns:
        df["weather_breaking_effect"] = (df["humidity"] / 100) * df["breaking_ball_pct"]
    return df

