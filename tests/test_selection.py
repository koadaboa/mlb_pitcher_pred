import pandas as pd

from src.features.selection import (
    _prune_feature_importance,
    select_features,
    filter_features_by_shap,
)


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


def test_select_features_pruning() -> None:
    df = pd.DataFrame(
        {
            "game_pk": [1, 2, 3, 4, 5],
            "x1_mean_3": [0, 1, 2, 3, 4],
            "x2_mean_3": [1, 1, 1, 1, 1],
            "strikeouts": [0, 1, 2, 3, 4],
        }
    )
    features_no_prune, _ = select_features(df, "strikeouts")
    assert set(features_no_prune) == {"x1_mean_3", "x2_mean_3"}
    features, _ = select_features(
        df,
        "strikeouts",
        prune_importance=True,
        importance_threshold=0.1,
    )
    assert features == ["x1_mean_3"]


def test_select_features_includes_base_numeric() -> None:
    """Numeric columns that aren't rolled should still be selected."""
    df = pd.DataFrame(
        {
            "game_pk": [1, 2, 3],
            "pitches": [80, 90, 100],
            "pitches_mean_3": [80, 85, 90],
            "strikeouts": [1, 2, 3],
        }
    )
    features, _ = select_features(df, "strikeouts")
    assert set(features) == {"pitches", "pitches_mean_3"}


def test_filter_features_by_shap(tmp_path) -> None:
    shap_path = tmp_path / "shap_importance.csv"
    pd.DataFrame({"feature": ["a", "b"], "importance": [0.1, 0.0]}).to_csv(shap_path, index=False)
    features = ["a", "b", "c"]
    filtered = filter_features_by_shap(features, shap_path=shap_path)
    assert filtered == ["a"]


def test_filter_features_by_shap_missing(tmp_path) -> None:
    features = ["a", "b"]
    filtered = filter_features_by_shap(features, shap_path=tmp_path / "missing.csv")
    assert filtered == features
