from __future__ import annotations

from typing import Mapping, Sequence, Tuple, Optional

import pandas as pd


def mean_target_encode(
    df: pd.DataFrame,
    columns: Sequence[str],
    target: Optional[str] = None,
    mapping: Optional[Mapping[str, pd.Series]] = None,
    *,
    suffix: str = "_enc",
) -> Tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Return dataframe with mean-encoded categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing ``columns`` and optionally ``target`` when ``mapping`` is
        ``None``.
    columns : Sequence[str]
        Categorical columns to encode.
    target : str | None
        Target variable used to compute means when fitting.
    mapping : Mapping[str, pd.Series] | None
        Precomputed encoding mapping. When provided, ``target`` is ignored and the
        mapping is applied to ``df``.
    suffix : str, default "_enc"
        Suffix for the encoded column names.
    """
    df = df.copy()
    enc_map: dict[str, pd.Series]
    if mapping is None:
        if target is None:
            raise ValueError("target must be provided when mapping is None")
        enc_map = {
            col: df.groupby(col)[target].mean() for col in columns
        }
    else:
        enc_map = dict(mapping)
    for col in columns:
        series = enc_map.get(col)
        if series is None:
            raise KeyError(f"Mapping for column '{col}' not found")
        df[f"{col}{suffix}"] = df[col].map(series).astype(float)
    return df, enc_map

