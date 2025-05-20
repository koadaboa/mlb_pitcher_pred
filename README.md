# MLB Pitcher Strikeout Prediction Model

This project aims to predict the number of strikeouts a Major League Baseball (MLB) pitcher will record in a given game using a combination of Statcast and MLB API data. The final goal is to create a highly performant machine learning model trained with LightGBM, suitable for applications like betting, fantasy sports, and general predictive analytics.

## Project Overview

* **Data Sources:**

  * `pybaseball` for Statcast data (pitch-level stats)
  * MLB Stats API for box scores and scheduling metadata
* **Data Pipeline:** Fully automated data-fetching and preprocessing pipeline
* **Current Focus:** Aggregation, feature engineering, and LightGBM model training

## Tables & Schemas

### `statcast_pitchers`

Contains granular pitch-level data from the pitcher's perspective. Columns:

```
['pitch_type', 'game_date', 'release_speed', 'release_pos_x', 'release_pos_z', 'player_name', 'batter', 'pitcher', 'events', 'description', 'spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated', 'zone', 'des', 'game_type', 'stand', 'p_throws', 'home_team', 'away_team', 'type', 'hit_location', 'bb_type', 'balls', 'strikes', 'game_year', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot', 'hc_x', 'hc_y', 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire', 'sv_id', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'hit_distance_sc', 'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate', 'release_extension', 'game_pk', 'fielder_2', 'fielder_3', 'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9', 'release_pos_y', 'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle', 'woba_value', 'woba_denom', 'babip_value', 'iso_value', 'launch_speed_angle', 'at_bat_number', 'pitch_number', 'pitch_name', 'home_score', 'away_score', 'bat_score', 'fld_score', 'post_away_score', 'post_home_score', 'post_bat_score', 'post_fld_score', 'if_fielding_alignment', 'of_fielding_alignment', 'spin_axis', 'delta_home_win_exp', 'delta_run_exp', 'bat_speed', 'swing_length', 'estimated_slg_using_speedangle', 'delta_pitcher_run_exp', 'hyper_speed', 'home_score_diff', 'bat_score_diff', 'home_win_exp', 'bat_win_exp', 'age_pit_legacy', 'age_bat_legacy', 'age_pit', 'age_bat', 'n_thruorder_pitcher', 'n_priorpa_thisgame_player_at_bat', 'pitcher_days_since_prev_game', 'batter_days_since_prev_game', 'pitcher_days_until_next_game', 'batter_days_until_next_game', 'api_break_z_with_gravity', 'api_break_x_arm', 'api_break_x_batter_in', 'arm_angle', 'pitcher_id', 'season']
```

### `statcast_batters`

Contains similar data to `statcast_pitchers` but from the batter's point of view. Columns:

```
['pitch_type', 'game_date', 'release_speed', 'release_pos_x', 'release_pos_z', 'player_name', 'batter', 'pitcher', 'events', 'description', 'spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated', 'zone', 'des', 'game_type', 'stand', 'p_throws', 'home_team', 'away_team', 'type', 'hit_location', 'bb_type', 'balls', 'strikes', 'game_year', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot', 'hc_x', 'hc_y', 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire', 'sv_id', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'hit_distance_sc', 'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate', 'release_extension', 'game_pk', 'fielder_2', 'fielder_3', 'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9', 'release_pos_y', 'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle', 'woba_value', 'woba_denom', 'babip_value', 'iso_value', 'launch_speed_angle', 'at_bat_number', 'pitch_number', 'pitch_name', 'home_score', 'away_score', 'bat_score', 'fld_score', 'post_away_score', 'post_home_score', 'post_bat_score', 'post_fld_score', 'if_fielding_alignment', 'of_fielding_alignment', 'spin_axis', 'delta_home_win_exp', 'delta_run_exp', 'bat_speed', 'swing_length', 'estimated_slg_using_speedangle', 'delta_pitcher_run_exp', 'hyper_speed', 'home_score_diff', 'bat_score_diff', 'home_win_exp', 'bat_win_exp', 'age_pit_legacy', 'age_bat_legacy', 'age_pit', 'age_bat', 'n_thruorder_pitcher', 'n_priorpa_thisgame_player_at_bat', 'pitcher_days_since_prev_game', 'batter_days_since_prev_game', 'pitcher_days_until_next_game', 'batter_days_until_next_game', 'api_break_z_with_gravity', 'api_break_x_arm', 'api_break_x_batter_in', 'arm_angle', 'season']
```

### `mlb_boxscores`

Includes metadata for each game such as:

```
['game_pk', 'game_date', 'away_team', 'home_team', 'game_number', 'double_header', 'away_pitcher_ids', 'home_pitcher_ids', 'hp_umpire', '1b_umpire', '2b_umpire', '3b_umpire', 'weather', 'temp', 'wind', 'elevation', 'dayNight', 'first_pitch', 'scraped_timestamp']
```

### `game_level_starting_pitchers`

Aggregated per-game stats for starting pitchers only. Rows are filtered from `statcast_pitchers` using the first pitch of each team in a game and requiring at least 3 innings pitched or 50 total pitches.

```text
['game_pk', 'pitcher_id', 'pitching_team', 'opponent_team', 'innings_pitched', 'pitches', 'strikeouts', 'swinging_strike_rate', 'first_pitch_strike_rate', 'fastball_pct', 'fastball_then_breaking_rate']
```
### `game_level_batters_vs_starters`

Aggregated per-game stats for each batter against their opponent's starting pitcher. Data is filtered from `statcast_batters` using pitches thrown only by the starting pitcher determined from `statcast_pitchers`.


```text
['game_pk', 'batter_id', 'pitcher_id', 'pitching_team', 'opponent_team', 'plate_appearances', 'at_bats', 'pitches', 'swings', 'whiffs', 'whiff_rate', 'called_strike_rate', 'strikeouts', 'strikeout_rate', 'strikeout_rate_behind', 'strikeout_rate_ahead', 'hits', 'singles', 'doubles', 'triples', 'home_runs', 'walks', 'hbp', 'avg', 'obp', 'slugging', 'ops', 'woba']
```

### `game_level_team_batting`

Aggregated per-game batting stats for each team facing the opponent's starting pitcher. Built from `game_level_batters_vs_starters` by summing the rows for all hitters on the same team.

```text
['game_pk', 'pitching_team', 'opponent_team', 'bat_plate_appearances', 'bat_at_bats', 'bat_pitches', 'bat_swings', 'bat_whiffs', 'bat_whiff_rate', 'bat_called_strike_rate', 'bat_strikeouts', 'bat_strikeout_rate', 'bat_strikeout_rate_behind', 'bat_strikeout_rate_ahead', 'bat_hits', 'bat_singles', 'bat_doubles', 'bat_triples', 'bat_home_runs', 'bat_walks', 'bat_hbp', 'bat_avg', 'bat_obp', 'bat_slugging', 'bat_ops', 'bat_woba']
```

### `game_level_matchup_stats`

Joins `game_level_starting_pitchers` with `game_level_team_batting` so each row represents one pitcher/team matchup for a game. Contains all pitcher metrics along with the aggregated opponent batting features.


## Pipeline Structure

1. **Data Fetching**

   * Pulls recent data from both Statcast and the MLB API
   * Stores raw data in SQLite tables

2. **Aggregation & Feature Engineering** (WIP)

   * Aggregate pitch-level data to game-level stats per pitcher
   * Feature examples: rolling averages, pitch mix %, rest days, weather, batter quality, etc.

3. **Model Training**

   * Predict target: strikeouts (`K`) in a game
   * Model: LightGBM with hyperparameter tuning
   * Evaluation metrics: MAE, RMSE, R^2

## Tools & Tech

* Python
* `pybaseball`, `requests` (for data fetching)
* SQLite (data storage)
* Pandas, NumPy (data processing)
* LightGBM (modeling)

## Next Steps

* Finalize feature aggregation logic
* Train baseline model and evaluate performance
* Add model monitoring & alerting for production use

## How to Contribute

Interested in contributing? Please open an issue or submit a PR.

## License

This project is currently private. Licensing terms TBD upon release.

---

Feel free to reach out with any questions or suggestions!
