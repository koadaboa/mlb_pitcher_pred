# MLB Pitcher Strikeout Prediction Model

This project aims to predict the number of strikeouts a Major League Baseball (MLB) pitcher will record in a given game using a combination of Statcast and MLB API data. The final goal is to create a highly performant machine learning model trained with LightGBM, suitable for applications like betting, fantasy sports, and general predictive analytics.

## Project Overview

* **Data Sources:**

  * `pybaseball` for Statcast data (pitch-level stats)
  * MLB Stats API for box scores and scheduling metadata
* **Data Pipeline:** Fully automated data-fetching and preprocessing pipeline
* **Multi-core Processing:** Aggregation scripts use all CPUs by default; override with `MAX_WORKERS` env var
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
### `game_starting_lineups`

Starting lineups pulled from the MLB boxscore endpoint.

```text
['game_pk', 'team', 'batter_id', 'batting_order', 'stand', 'catcher_id']
```


### `game_level_starting_pitchers`

Aggregated per-game stats for starting pitchers only. Rows are filtered from `statcast_pitchers` using the first pitch of each team in a game and requiring at least 3 innings pitched or 50 total pitches. The table now includes pitch usage percentages and whiff rates for common pitch types.

```text
['game_pk', 'pitcher_id', 'pitching_team', 'opponent_team', 'innings_pitched',
 'pitches', 'strikeouts', 'swinging_strike_rate', 'first_pitch_strike_rate',
 'fastball_pct', 'slider_pct', 'curve_pct', 'changeup_pct', 'cutter_pct',
 'sinker_pct', 'splitter_pct', 'fastball_whiff_rate', 'slider_whiff_rate',
 'curve_whiff_rate', 'changeup_whiff_rate', 'cutter_whiff_rate',
 'sinker_whiff_rate', 'splitter_whiff_rate', 'fastball_then_breaking_rate']
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

Joins `game_level_starting_pitchers` with `game_level_team_batting` so each row represents one pitcher/team matchup for a game. Contains all pitcher metrics along with the aggregated opponent batting features. Built by running `python -m src.create_pitcher_vs_team_stats`.
### `rolling_pitcher_features`

Derived from `game_level_starting_pitchers`, this table contains rolling-window
statistics for each pitcher. For every numeric metric we compute prior-game
means and standard deviations over the window sizes defined in
`StrikeoutModelConfig.WINDOW_SIZES`. Momentum features capture the difference
between the current game value and the previous rolling mean.
All calculations use a one-game shift to avoid any data that occurs after the
game begins.

### `rolling_pitcher_vs_team`

Rolling-window statistics for each pitcher against a specific opponent. These
features are computed from `game_level_matchup_details` and capture how a
pitcher has performed historically versus that team.

### `contextual_features`

Adds rolling averages for game context variables. Umpire- and weather-specific
trends are aggregated alongside stadium information based on the home team. Raw
weather values such as temperature, wind speed and park elevation are also
included for each game.

### Required Tables

Training expects the following tables to be present in the SQLite database:

- Raw data: `statcast_pitchers`, `statcast_batters`, `mlb_boxscores`
- Aggregations: `game_level_starting_pitchers`, `game_level_batters_vs_starters`,
  `game_level_team_batting`, `game_level_matchup_stats`
- Rolling features: `rolling_pitcher_features`, `rolling_pitcher_vs_team`,
  `contextual_features`, `lineup_trends`
- Final matrix: `model_features`

All of these can be generated by running `python -m src.scripts.run_feature_engineering`.

## Pipeline Structure


1. **Data Fetching**

    * Run `python -m src.scripts.data_fetcher` to download Statcast and MLB API data
    * Populates the raw tables listed above

2. **Aggregation & Feature Engineering**

    * Run `python -m src.scripts.run_table_creation` to build the initial
      aggregated tables like starting pitcher stats, batter splits, team batting
      metrics and catcher defense ratings.

    * Execute `python -m src.scripts.run_feature_engineering` to build all aggregated tables
      by running `engineer_pitcher_features`, `engineer_workload_features`,
      `engineer_opponent_features`, `engineer_contextual_features`,
      `engineer_lineup_trends` and finally `build_model_features`.
    * The `lineup_trends` table is required for the final join when creating
      `model_features`.

3. **Hyperparameter Tuning**

    * Search model parameters with `tune_lightgbm.py`, `tune_xgb.py` or `tune_catboost.py`
    * Each script saves a JSON file in `models/` containing the best parameters

4. **Model Training**

    * Predict target: strikeouts (`K`) in a game
    * Train using LightGBM, XGBoost or CatBoost
    * Evaluation metrics: MAE, RMSE, R^2

## Tools & Tech

* Python 3.10+
* `pybaseball`, `requests` (for data fetching)
* SQLite (data storage)
* Pandas, NumPy (data processing)
* LightGBM (modeling)
* XGBoost (alternate modeling)

### Logging

Logs are written to the `logs/` directory using the helper in `src.utils`.
`create_starting_pitcher_table.py` now reports how many potential starters were
found and prints progress every 100 games to `logs/create_starting_pitcher_table.log`.

The script leverages multiple CPU cores to process games in parallel,
dramatically reducing runtime on multi-core machines.

### Multi-core Usage

`create_starting_pitcher_table.py` launches one process per CPU by default.
The worker count is controlled by `DataConfig.MAX_WORKERS`, which can be
overridden via the `MAX_WORKERS` environment variable:

```bash
MAX_WORKERS=4 python -m src.create_starting_pitcher_table
```

### Running Feature Engineering

Execute all feature builders and produce the `model_features` table. The command
sequentially calls `engineer_pitcher_features`, `engineer_workload_features`,
`engineer_opponent_features`, `engineer_contextual_features`,
`engineer_lineup_trends` and then `build_model_features`. The `lineup_trends`
table must be present for the final
join:

```bash
python -m src.scripts.run_feature_engineering --db-path path/to/pitcher_stats.db
```

Limit processing to a single season with `--year`:

```bash
python -m src.scripts.run_feature_engineering --db-path path/to/pitcher_stats.db --year 2024
```

If the feature tables already exist, the script only processes games newer than
the latest date stored in each table and appends the new rows.

Use the `--n-jobs` option to control how many processes are used when computing
rolling features:

```bash
python -m src.scripts.run_feature_engineering --db-path path/to/pitcher_stats.db --n-jobs 8
```

### Data Leakage Prevention

`build_model_features` removes columns that could leak target information. Raw
game outcome stats (e.g. `bat_strikeouts`) and identifier fields such as
`away_pitcher_ids`, `home_pitcher_ids`, and `scraped_timestamp` are dropped
before saving `model_features`. Only rolling statistics or whitelisted numeric
columns like `temp`, `wind_speed`, `park_factor`, and `team_k_rate` are
retained.

### Hyperparameter Tuning & Training

Run Optuna tuning for each model to find the best parameters:

```bash
python -m src.tune_lightgbm --db-path path/to/pitcher_stats.db --trials 50
python -m src.tune_xgb --db-path path/to/pitcher_stats.db --trials 50
python -m src.tune_catboost --db-path path/to/pitcher_stats.db --trials 50
```

Each script writes a JSON file such as `models/lgbm_best_params.json` (and similar files for XGBoost and CatBoost) containing the optimal hyperparameters. To train using these values:

```python
import json
from src.config import StrikeoutModelConfig
from src.train_model import load_dataset, split_by_year, train_lgbm

with open("models/lgbm_best_params.json") as f:
    StrikeoutModelConfig.LGBM_BASE_PARAMS.update(json.load(f))

df = load_dataset()
train_df, test_df = split_by_year(df)
train_lgbm(train_df, test_df)
```

## Next Steps

* Train baseline model and evaluate performance

```bash
python -m src.scripts.train_baseline_model --db-path path/to/pitcher_stats.db
```

This computes a 5-fold time-series CV RMSE and evaluates on the test set.
* Compute SHAP feature importances

```bash
python -m src.scripts.shap_sweep --db-path path/to/pitcher_stats.db
```

This writes `plots/shap_importance.csv` with the mean absolute SHAP value for each feature.
* Add model monitoring & alerting for production use

## How to Contribute

Interested in contributing? Please open an issue or submit a PR.

## License

This project is currently private. Licensing terms TBD upon release.

---

Feel free to reach out with any questions or suggestions!
