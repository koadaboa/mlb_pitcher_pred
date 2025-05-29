import pandas as pd

from src.features.selection import _prune_feature_importance


def test_prune_feature_importance() -> None:
    df = pd.DataFrame({"x1": [0, 1, 2, 3, 4], "x2": [1, 1, 1, 1, 1]})
    target = pd.Series([0, 1, 2, 3, 4])
    cols, imp = _prune_feature_importance(df, target, threshold=0.1)
    assert cols == ["x1"]
    assert imp.loc["x1"] > imp.loc["x2"]


def test_prune_feature_importance_lightgbm() -> None:
    df = pd.DataFrame({"x1": [0, 1, 2, 3, 4], "x2": [1, 1, 1, 1, 1]})
    target = pd.Series([0, 1, 2, 3, 4])
    cols, _ = _prune_feature_importance(
        df,
        target,
        threshold=0.1,
        method="lightgbm",
    )
    assert cols == ["x1"]
