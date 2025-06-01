import pandas as pd
import numpy as np


def recent_average_predict(train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            pitcher_col: str = "pitcher_id",
                            date_col: str = "game_date",
                            target_col: str = "strikeouts",
                            window: int = 5) -> pd.Series:
    """Predict strikeouts using the mean of a pitcher's previous games.

    Parameters
    ----------
    train_df : pd.DataFrame
        Historical training data containing the target column.
    test_df : pd.DataFrame
        Test data for which predictions are required.
    pitcher_col : str
        Column identifying the pitcher.
    date_col : str
        Column with game dates.
    target_col : str
        Column with strikeout totals.
    window : int
        Number of prior games to average.

    Returns
    -------
    pd.Series
        Baseline predictions aligned to ``test_df`` index.
    """
    if train_df is None or test_df is None:
        return pd.Series(dtype=float)

    # Work on copies to avoid modifying original frames
    train_df = train_df[[pitcher_col, date_col, target_col]].copy()
    test_df = test_df[[pitcher_col, date_col, target_col]].copy()

    train_df["__dataset__"] = "train"
    test_df["__dataset__"] = "test"
    train_df["_orig_index"] = train_df.index
    test_df["_orig_index"] = test_df.index

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined[date_col] = pd.to_datetime(combined[date_col])
    combined = combined.sort_values([pitcher_col, date_col])

    rolling = (
        combined.groupby(pitcher_col)[target_col]
        .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    combined["baseline_pred"] = rolling

    preds = (
        combined[combined["__dataset__"] == "test"]
        .set_index("_orig_index")["baseline_pred"]
        .sort_index()
    )
    return preds

