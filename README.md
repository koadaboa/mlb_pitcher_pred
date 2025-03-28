# Pitcher Strikeout Prediction - Optimizing for 1-Strikeout Precision

This guide outlines the process for optimizing ML models to achieve high-precision strikeout predictions (within 1 strikeout) and creating ensembles to further improve accuracy.

## 1. Model Optimization for 1-Strikeout Precision

The `optimize_models.py` script uses Bayesian optimization via Optuna to find the optimal hyperparameters specifically targeting the "within 1 strikeout" metric.

### Usage

```bash
# Optimize for Within 1 Strikeout accuracy (recommended)
python -m src.scripts.optimize_models --model xgboost --trials 100 --metric within_1_strikeout

# Optimize all models with focus on 1-strikeout precision
python -m src.scripts.optimize_models --model all --metric within_1_strikeout --trials 50
```

### Optimization Metrics:
- `within_1_strikeout`: Percentage of predictions within 1 strikeout of actual value (primary target)
- `over_under_accuracy`: Accuracy in predicting over/under against the average
- `neg_mean_squared_error`: Negative RMSE (for minimization)
- `neg_mean_absolute_error`: Negative MAE (for minimization)

### Optimization Outputs

The script creates a timestamped directory in `models/optimization/{model_type}_{timestamp}/` containing:
- Optimization database (`optimization.db`)
- The best model (`optimized_{model_type}_model.pkl`)
- Visualizations:
  - Parameter importance
  - Optimization history
  - Betting metrics progression

## 2. Creating High-Precision Ensembles

After optimizing individual models, use the `create_ensemble.py` script to build and evaluate different ensemble methods aimed at maximizing 1-strikeout precision.

### Usage

```bash
# Create ensembles using optimized models
python -m src.scripts.create_ensemble \
  --model-paths models/optimization/*/optimized_*_model.pkl
```

### Ensemble methods

The script evaluates several ensemble approaches:
1. **Weighted averaging** with different weighting methods:
   - `betting_accuracy`: Weights based on Within-1-Strikeout accuracy (recommended)
   - `inverse_error`: Weights inversely proportional to RMSE
   - `equal`: Equal weights for all models

2. **Stacking**: Uses a meta-model (Ridge regression) to combine base model predictions

### Ensemble outputs

The script generates a comprehensive comparison in the `models/ensemble/` directory:
- Visualizations of model performance across various metrics
- Rankings of all models (base and ensemble)
- The best ensemble model is saved as `best_ensemble_model.pkl`
- A standardized copy is saved as `models/strikeout_ensemble_model.pkl`

## 3. Evaluating Model Performance 

To see detailed metrics including the critical "Within 1 Strikeout" accuracy:

```bash
python -m src.scripts.model_comparison
```

This will print a summary table with key metrics for all models, including:
- RMSE and MAE values
- Within 1 Strikeout accuracy percentage
- Generate detailed comparison visualizations

## Strategies for Maximizing 1-Strikeout Accuracy

Achieving high precision (within 1 strikeout) requires several specialized approaches:

1. **Feature Engineering**:
   - Create pitcher consistency metrics (standard deviation over different periods)
   - Add matchup-specific features
   - Include more granular rest day impact metrics

2. **Model Specialization**:
   - Consider separate models for different pitcher types (starters vs. relievers)
   - Build models specialized for different strikeout ranges

3. **Ensemble Refinement**:
   - Use confidence-weighted predictions
   - Create ensemble rules that optimize specifically for within-1-strikeout accuracy
   - Implement model stacking with a meta-model trained specifically on the "within 1 strikeout" objective

## Key Hyperparameters For 1-Strikeout Precision

When optimizing specifically for 1-strikeout precision, focus on these parameters:

### Random Forest
- `min_samples_leaf`: Critical for reducing variance (3-7 typically works well)
- `max_depth`: Prevents overfitting (15-25 range often optimal)
- `n_estimators`: Higher values (200-500) improve precision

### XGBoost
- `learning_rate`: Lower values (0.01-0.05) tend to improve precision
- `min_child_weight`: Important for preventing overfitting to noise
- `gamma`: Controls precision by requiring minimum loss reduction
- `subsample`: Values around 0.8-0.9 can improve stability

### LightGBM
- `num_leaves`: Lower than default for better generalization
- `min_data_in_leaf`: Key parameter for precision (30-60 often works well)
- `feature_fraction`: Reduce to 0.7-0.8 to limit model complexity
- `bagging_fraction`: Values around 0.8 help with consistency

## Workflow Summary

1. **Optimize for Within-1-Strikeout**: Run optimization with this specific metric
2. **Create Specialized Ensembles**: Combine models with weighting toward 1-strikeout accuracy
3. **Evaluate and Refine**: Use model_comparison.py to track improvements in precision
4. **Iterative Improvement**: Continue to enhance features specifically targeting precision