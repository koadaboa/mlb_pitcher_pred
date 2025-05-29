import pandas as pd
import numpy as np

from src.features.selection import _calculate_vif, _prune_feature_importance


def test_calculate_vif_with_nan_and_inf() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, np.inf],
            "b": [np.inf, 1.0, 2.0, np.nan],
        }
    )
    result = _calculate_vif(df)
    assert list(result.index) == ["a", "b"]
    assert result.isna().all()


def test_prune_feature_importance() -> None:
    df = pd.DataFrame({"x1": [0, 1, 2, 3, 4], "x2": [1, 1, 1, 1, 1]})
    target = pd.Series([0, 1, 2, 3, 4])
    cols, imp = _prune_feature_importance(df, target, threshold=0.1)
    assert cols == ["x1"]
    assert imp.loc["x1"] > imp.loc["x2"]

