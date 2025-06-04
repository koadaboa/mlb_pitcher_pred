"""Utilities for grouping related feature names."""

from __future__ import annotations

PREFIX_TO_GROUP = {
    # Opponent related metrics
    "opp_": "Opponent",
    "team_": "Opponent",
    "team_hand_": "Opponent",
    "lineup_": "Opponent",
    "slot": "Opponent",
    "opp_batter_": "Opponent",
    "opp_lineup_": "Opponent",
    "batter_": "Opponent",
    # Context and environmental factors
    "wx_": "Context",
    "venue_": "Context",
    "ump_": "Context",
    "catcher_": "Context",
    "temp": "Context",
    "wind_speed": "Context",
    "humidity": "Context",
    "elevation": "Context",
    "park_factor": "Context",
    "day_of_week": "Context",
    "travel_distance": "Context",
    "pitches_last_": "Context",
    "rest_days": "Context",
    "on_il": "Context",
    "days_since_il": "Context",
    # Pitch quality and pitch mix
    "fastball_": "Pitch_Quality",
    "slider_": "Pitch_Quality",
    "curve_": "Pitch_Quality",
    "changeup_": "Pitch_Quality",
    "cutter_": "Pitch_Quality",
    "sinker_": "Pitch_Quality",
    "splitter_": "Pitch_Quality",
    "pfx_": "Pitch_Quality",
    "release_": "Pitch_Quality",
    "plate_": "Pitch_Quality",
    "spin": "Pitch_Quality",
    "whiff": "Pitch_Quality",
    "csw": "Pitch_Quality",
    "zone_": "Pitch_Quality",
    "chase_": "Pitch_Quality",
    "fastball_then_": "Pitch_Quality",
    "offspeed_to_": "Pitch_Quality",
    "unique_pitch_types": "Pitch_Quality",
}

# Some columns do not share common prefixes but should still be categorized
SPECIFIC_GROUPS = {
    "pitches_last_7d": "Context",
    "rest_days": "Context",
    "on_il": "Context",
    "days_since_il": "Context",
    "day_of_week": "Context",
    "travel_distance": "Context",
    "days_into_season": "Context",
    "month_bucket": "Context",
}

DEFAULT_GROUP = "Performance"


def assign_feature_group(feature: str) -> str:
    """Return the feature group for ``feature`` based on prefix rules."""
    if feature in SPECIFIC_GROUPS:
        return SPECIFIC_GROUPS[feature]
    for prefix, group in PREFIX_TO_GROUP.items():
        if feature.startswith(prefix):
            return group
    return DEFAULT_GROUP

