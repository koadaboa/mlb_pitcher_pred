# Pitcher Strikeout Prediction - Model Optimization and Ensembling

This guide outlines the process for optimizing individual ML models and creating an ensemble for pitcher strikeout prediction.

## 1. Model Optimization

The `optimize_models.py` script uses Bayesian optimization via Optuna to find the best hyperparameters for each model type.

### Usage

```bash
# Optimize a specific model (rf, xgboost, or lightgbm)
python -m src.scripts.optimize_models --model xgboost --trials 100 --metric within_2_strikeouts

# Optimize all models with default settings
python -m src.scripts.optimize_models --model all

# Customize training years and optimization metric
python -m src.scripts.optimize_models --model lightgbm --years 2019 2021 2022 --metric over_under_accuracy
```

### Available optimization metrics:
- `within_2_strikeouts`: Percentage of predictions within 2 strikeouts of actual value (default)
- `over_under_accuracy`: Accuracy in predicting over/under against average
- `neg_mean_squared_error`: Negative RMSE (for minimization)
- `neg_mean_absolute_error`: Negative MAE (for minimization)
- `r2`: R-squared score

### Optimization outputs

The script creates a timestamped directory in `models/optimization/{model_type}_{timestamp}/` containing:
- Optimization database (`optimization.db`)
- The best model (`optimized_{model_type}_model.pkl`)
- Visualizations:
  - Parameter importance
  - Optimization history
  - Parallel coordinate plot
  - Betting metrics progression

## 2. Creating Ensembles

After optimizing individual models, use the `create_ensemble.py` script to build and evaluate different ensemble methods.

### Usage

```bash
# Create ensembles using optimized models
python -m src.scripts.create_ensemble \
  --model-paths models/optimization/rf_*/optimized_rf_model.pkl \
              models/optimization/xgboost_*/optimized_xgboost_model.pkl \
              models/optimization/lightgbm_*/optimized_lightgbm_model.pkl \
  --test-years 2023 2024
```

### Ensemble methods

The script automatically evaluates several ensemble approaches:
1. **Weighted averaging** with different weighting methods:
   - `equal`: Equal weights for all models
   - `inverse_error`: Weights inversely proportional to RMSE
   - `betting_accuracy`: Weights based on betting-relevant metrics

2. **Stacking**: Uses a meta-model (Ridge regression) to combine base model predictions

### Ensemble outputs

The script generates a comprehensive comparison in the `models/ensemble/` directory:
- Visualizations of model performance across various metrics
- Rankings of all models (base and ensemble)
- The best ensemble model is saved as `best_ensemble_model.pkl`
- A standardized copy is saved as `models/strikeout_ensemble_model.pkl`

## 3. Re-running Model Comparison

After creating an optimized ensemble, use the existing model comparison script to evaluate it against the base models:

```bash
python -m src.scripts.model_comparison
```

## Key hyperparameters being optimized

### Random Forest
- `n_estimators`: Number of trees (50-500)
- `max_depth`: Maximum tree depth (5-30)
- `min_samples_split`: Minimum samples required to split (2-20)
- `min_samples_leaf`: Minimum samples in leaf nodes (1-10)
- `max_features`: Feature subset strategy ('sqrt', 'log2', or None)
- `bootstrap`: Whether to use bootstrap samples (True/False)

### XGBoost
- `n_estimators`: Number of boosting rounds (50-500)
- `max_depth`: Maximum tree depth (3-12)
- `learning_rate`: Step size shrinkage (0.01-0.3)
- `subsample`: Subsample ratio of training instances (0.5-1.0)
- `colsample_bytree`: Subsample ratio of columns (0.5-1.0)
- `min_child_weight`: Minimum sum of instance weight needed in a child (1-10)
- `gamma`: Minimum loss reduction for split (0-5)
- `reg_alpha`: L1 regularization (0-5)
- `reg_lambda`: L2 regularization (0-5)

### LightGBM
- `n_estimators`: Number of boosting rounds (50-500)
- `learning_rate`: Step size shrinkage (0.01-0.3)
- `num_leaves`: Maximum number of leaves in one tree (20-150)
- `max_depth`: Maximum tree depth (3-12)
- `min_data_in_leaf`: Minimum data in one leaf (10-100)
- `feature_fraction`: Feature subset ratio (0.5-1.0)
- `bagging_fraction`: Bagging fraction (0.5-1.0)
- `bagging_freq`: Bagging frequency (1-10)
- `min_gain_to_split`: Minimum gain to perform split (0-5)
- `lambda_l1`: L1 regularization (0-5)
- `lambda_l2`: L2 regularization (0-5)

## Workflow Summary

1. **Optimize individual models**: Run the optimization script for each model type
2. **Create and evaluate ensembles**: Combine the optimized models into various ensemble methods
3. **Compare all models**: Run the model comparison script to see how the ensemble performs
4. **Use the best model**: The best model is saved with a standardized name that can be used with your prediction pipeline