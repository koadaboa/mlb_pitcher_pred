# AGENT.md

## 📝 Project Overview
This repository contains a machine learning pipeline for predicting MLB starting pitcher strikeout totals for a given game. Data is fetched from Statcast using `pybaseball` and the MLB API. The model is trained on historical pitch-level data, engineered into game-level aggregates, and optimized using LightGBM.

## 📁 Important Files
- `src/create_starting_pitcher_table.py`: Aggregates pitcher-level stats per game.
- `src/create_batter_vs_starter_table.py`: Aggregates batter performance against the starter.
- `src/create_pitcher_vs_team_stats.py`: Joins per-game pitcher stats with
  aggregated opponent batting metrics to produce `game_level_matchup_stats`.
- `src/features.py`: Feature engineering logic (rolling averages, ratios, etc.)
- `src/train_model.py`: Training pipeline using LightGBM, with evaluation metrics.
- `data/`: Contains raw and processed SQLite databases.
- `notebooks/`: Experimental and analysis notebooks.

## 🎯 Primary Tasks
- Engineer reliable features from Statcast and advanced boxscore data.
- Identify and validate probable starting pitchers.
- Train a strikeout prediction model using LightGBM.
- Evaluate performance with RMSE, MAE, and a custom `within_1_so` accuracy metric.

## 🚫 Avoid
- Using rolling stats in aggregation scripts (those belong in `features.py`).
- Including relief pitchers or starting pitchers in a bullpen game in training data.
- Heavy I/O in training — prefer using pre-aggregated data from SQLite if that works better.
- Data leakage by any means necessary, use shift if necessary to avoid data leakage.

## 📊 Data Sources
- `statcast_pitchers`: Pitch-level data via `pybaseball.statcast()`.
- `mlb_api`: Used to validate game-level metadata and player statuses.
- Stored in a 19GB SQLite database, synced to S3 for remote access.

## 🔧 Tools & Libraries
- Python 3.10+
- pandas, numpy, scikit-learn, lightgbm, sqlite3, pybaseball
- Codex agent is allowed to run shell commands, edit code files, and use Git.

## 🧠 Suggestions
- When dealing with NaNs, assume missing values are due to game context (e.g. no runners = empty `on_2b`).
- When in doubt about the starter, cross-reference with pitch count > 25 and `inning == 1`.