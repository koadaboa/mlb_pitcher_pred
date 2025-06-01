"""Feature engineering entry points."""

from .engineer_features import engineer_pitcher_features
from .workload_features import engineer_workload_features
from .contextual import (
    engineer_opponent_features,
    engineer_contextual_features,
    engineer_lineup_trends,
    engineer_catcher_defense,
)
from .join import build_model_features
from .encoding import mean_target_encode

__all__ = [
    "engineer_pitcher_features",
    "engineer_workload_features",
    "engineer_opponent_features",
    "engineer_contextual_features",
    "engineer_lineup_trends",
    "engineer_catcher_defense",
    "build_model_features",
    "mean_target_encode",
]
