"""Feature selection utilities used during model training.

All numeric columns are considered as candidate features unless explicitly
excluded. When requested via arguments, low-importance features can be
pruned using tree-based model importances.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import re
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor

import pandas as pd
from src.config import StrikeoutModelConfig

# Columns that should never be used as model features
BASE_EXCLUDE_COLS: List[str] = [
    "game_pk",
    "game_date",
    "pitcher_id",
    "pitching_team",
    "opponent_team",
]

def _prune_feature_importance(
    df: pd.DataFrame,
    target: pd.Series,
    threshold: float,
    *,
    method: str = "extra_trees",
) -> Tuple[List[str], pd.Series]:
    """Drop columns with low feature importance.

    Parameters
    ----------
    df : pd.DataFrame
        Training features.
    target : pd.Series
        Target values.
    threshold : float
        Importance threshold relative to the maximum importance.
    method : str, optional
        ``"extra_trees"`` (default) or ``"lightgbm"`` to determine how
        feature importance is calculated.
    """

    if df.empty:
        return [], pd.Series(dtype=float)

    if method == "lightgbm":
        model = LGBMRegressor(
            n_estimators=100,
            random_state=StrikeoutModelConfig.RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(df, target)
        importances = pd.Series(model.feature_importances_, index=df.columns)
    else:
        model = ExtraTreesRegressor(
            n_estimators=50,
            random_state=StrikeoutModelConfig.RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(df, target)
        importances = pd.Series(model.feature_importances_, index=df.columns)

    keep_mask = importances >= (threshold * importances.max())
    return df.columns[keep_mask].tolist(), importances






def select_features(
    df: pd.DataFrame,
    target_variable: str,
    exclude_cols: Optional[Iterable[str]] = None,
    *,
    prune_importance: bool = False,
    importance_threshold: float = 0.01,
    importance_method: str = "extra_trees",
) -> Tuple[List[str], pd.DataFrame]:
    """Return a list of selected features and an optional info DataFrame.

    When ``prune_importance`` is ``True``, ``importance_method`` controls
    whether Extra Trees or LightGBM feature importances are used.
    """

    exclude_set = set(BASE_EXCLUDE_COLS)
    if exclude_cols:
        exclude_set.update(exclude_cols)
    exclude_set.add(target_variable)

    pattern = re.compile(r"_(?:mean|std|momentum)_\d+$")
    allowed_numeric = set(StrikeoutModelConfig.ALLOWED_BASE_NUMERIC_COLS)

    numeric_cols = [
        c
        for c in df.columns
        if c not in exclude_set
        and pd.api.types.is_numeric_dtype(df[c])
        and (pattern.search(c) or c in allowed_numeric)
    ]
    selected = numeric_cols

    info_df = pd.DataFrame()
    if prune_importance and numeric_cols:
        selected, imp = _prune_feature_importance(
            df[selected],
            df[target_variable],
            importance_threshold,
            method=importance_method,
        )
        info_df = imp.rename("importance").to_frame()

    return selected, info_df
