# Pitcher Strikeout Prediction - Optimizing for 1-Strikeout Precision

This guide outlines the process for optimizing LightGBM models to achieve high-precision strikeout predictions (within 1 strikeout).

## 1. Model Optimization for 1-Strikeout Precision

The `optimize_models.py` script uses Bayesian optimization via Optuna to find the optimal hyperparameters specifically targeting the "within 1 strikeout" metric.

### Usage

```bash
# Optimize for Within 1 Strikeout accuracy (recommended)
python -m src.scripts.optimize_models --metric within_1_strikeout --trials 100

# Optimize with custom training years
python -m src.scripts.optimize_models --years 2019 2021 2022 --metric within_1_strikeout --trials 100