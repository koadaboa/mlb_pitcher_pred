"""Feature engineering entry points."""

from .engineer_features import engineer_pitcher_features
from .batter_pitcher_history import engineer_batter_pitcher_history
from .contextual import (
    engineer_opponent_features,
    engineer_contextual_features,
    engineer_lineup_trends,
)
from .join import build_model_features
from .encoding import mean_target_encode

__all__ = [
    "engineer_pitcher_features",
    "engineer_opponent_features",
    "engineer_contextual_features",
    "engineer_lineup_trends",
    "engineer_batter_pitcher_history",
    "build_model_features",
    "mean_target_encode",
]
