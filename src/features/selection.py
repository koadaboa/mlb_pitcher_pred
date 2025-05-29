"""Feature selection utilities used during model training.

This module exposes a simple interface for selecting model features. The
current implementation keeps the logic lightweight: unless requested via
arguments, all numeric columns are used. Optional Variance Inflation
Factor (VIF) and LightGBM SHAP pruning can be applied when desired.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import re
from sklearn.ensemble import ExtraTreesRegressor

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.config import StrikeoutModelConfig

# Columns that should never be used as model features
BASE_EXCLUDE_COLS: List[str] = [
    "game_pk",
    "game_date",
    "pitcher_id",
    "pitching_team",
    "opponent_team",
]


def _calculate_vif(df: pd.DataFrame) -> pd.Series:
    """Return VIF for each column in ``df``."""
    if df.empty:
        return pd.Series(dtype=float)

    # Replace infinities then drop any rows containing NaN values
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_df.empty:
        return pd.Series([np.nan] * len(df.columns), index=df.columns)

    X = clean_df.assign(const=1)
    vifs = [
        variance_inflation_factor(X.values, i) for i in range(len(clean_df.columns))
    ]
    series = pd.Series(vifs, index=clean_df.columns)
    return series.reindex(df.columns)

def _prune_feature_importance(
    df: pd.DataFrame, target: pd.Series, threshold: float
) -> Tuple[List[str], pd.Series]:
    """Drop columns with low feature importance using ExtraTreesRegressor."""
    if df.empty:
        return [], pd.Series(dtype=float)
    model = ExtraTreesRegressor(
        n_estimators=50,
        random_state=StrikeoutModelConfig.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(df, target)
    importances = pd.Series(model.feature_importances_, index=df.columns)
    keep_mask = importances >= (threshold * importances.max())
    return df.columns[keep_mask].tolist(), importances



def _prune_vif(df: pd.DataFrame, threshold: float) -> List[str]:
    """Iteratively drop columns with VIF greater than ``threshold``."""
    cols = df.columns.tolist()
    while True:
        vif = _calculate_vif(df[cols])
        if vif.empty or vif.dropna().empty:
            break
        max_vif = vif.max()
        if max_vif <= threshold:
            break
        drop_col = vif.idxmax()
        cols.remove(drop_col)
    return cols


def _prune_shap(
    df: pd.DataFrame,
    model,
    threshold: float,
    sample_frac: float,
) -> List[str]:
    """Return columns with mean absolute SHAP value >= ``threshold``."""
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=0)
    shap_values = model.predict(df, pred_contrib=True)
    # Exclude the bias term (last column)
    shap_vals = np.abs(shap_values[:, :-1])
    mean_importance = shap_vals.mean(axis=0)
    keep_mask = mean_importance >= (threshold * mean_importance.max())
    return df.columns[keep_mask].tolist()


def select_features(
    df: pd.DataFrame,
    target_variable: str,
    exclude_cols: Optional[Iterable[str]] = None,
    *,
    prune_importance: bool = False,
    importance_threshold: float = 0.01,
    prune_vif: bool = False,
    vif_threshold: float = 5.0,
    prune_shap: bool = False,
    shap_model=None,
    shap_threshold: float = 0.01,
    shap_sample_frac: float = 1.0,
) -> Tuple[List[str], pd.DataFrame]:
    """Return a list of selected features and an optional info DataFrame."""

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
            df[selected], df[target_variable], importance_threshold
        )
        info_df = imp.rename("importance").to_frame()

    if prune_vif and selected:
        selected = _prune_vif(df[selected], vif_threshold)
        info_df = _calculate_vif(df[selected]).rename("vif").to_frame()

    if prune_shap and shap_model is not None and selected:
        keep = _prune_shap(df[selected], shap_model, shap_threshold, shap_sample_frac)
        info_df = info_df.reindex(keep)
        selected = keep

    return selected, info_df
