import pandas as pd
import pytest

from src.config import StrikeoutModelConfig
from src.train_model import train_lgbm
from src.train_catboost import train_catboost
from src.train_xgb_model import train_xgb
from src.train_ensemble_model import train_ensemble


def _make_train_test_dfs(n_rows: int = 20):
    df = pd.DataFrame({
        "game_pk": range(n_rows),
        "game_date": pd.date_range("2024-01-01", periods=n_rows, freq="D"),
        "pitcher_id": [1] * n_rows,
        "pitching_team": ["A"] * n_rows,
        "opponent_team": ["B"] * n_rows,
        "x1_mean_3": range(n_rows),
        "x2_mean_3": range(n_rows),
        "temp": range(n_rows),
        "home_team": ["A", "B"] * (n_rows // 2),
        "strikeouts": range(n_rows),
    })
    half = n_rows // 2
    return df.iloc[:half], df.iloc[half:]


def test_train_lgbm_runs(monkeypatch):
    pytest.importorskip("lightgbm")
    train_df, test_df = _make_train_test_dfs()
    monkeypatch.setattr(StrikeoutModelConfig, "FINAL_ESTIMATORS", 10)
    monkeypatch.setattr(StrikeoutModelConfig, "EARLY_STOPPING_ROUNDS", 2)
    model, metrics, features = train_lgbm(train_df, test_df)
    assert hasattr(model, "predict")
    assert set(metrics) == {"rmse", "mae", "within_1_so"}


def test_train_catboost_runs(monkeypatch):
    pytest.importorskip("catboost")
    train_df, test_df = _make_train_test_dfs()
    monkeypatch.setattr(StrikeoutModelConfig, "FINAL_ESTIMATORS", 10)
    model, metrics = train_catboost(train_df, test_df)
    assert hasattr(model, "predict")
    assert set(metrics) == {"rmse", "mae", "within_1_so"}


def test_train_xgb_runs(monkeypatch):
    pytest.importorskip("xgboost")
    train_df, test_df = _make_train_test_dfs()
    monkeypatch.setattr(StrikeoutModelConfig, "FINAL_ESTIMATORS", 10)
    model, metrics = train_xgb(train_df, test_df)
    assert hasattr(model, "predict")
    assert set(metrics) == {"rmse", "mae", "within_1_so"}


def test_train_ensemble_runs(monkeypatch):
    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    pytest.importorskip("catboost")
    train_df, test_df = _make_train_test_dfs()
    monkeypatch.setattr(StrikeoutModelConfig, "FINAL_ESTIMATORS", 10)
    model, metrics = train_ensemble(train_df, test_df)
    assert hasattr(model.meta_model, "predict")
    assert set(metrics) == {"rmse", "mae", "within_1_so"}
