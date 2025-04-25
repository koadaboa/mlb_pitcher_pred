import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try imports for pruning
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError:
    variance_inflation_factor = None

try:
    import shap
except ImportError:
    shap = None

# Exclude list of known leakage or non-features
BASE_EXCLUDE_COLS = [
    'index', '', 'pitcher_id', 'player_name', 'game_pk', 'home_team', 'away_team', 'opponent',
    'opponent_team_name', 'game_date', 'season', 'game_month', 'year',
    'p_throws', 'stand', 'team', 'Team', 'opp_base_team', 'opp_adv_team',
    'opp_adv_opponent', 'ballpark', 'home_plate_umpire', 'strikeouts',
    'batters_faced','total_pitches','innings_pitched','avg_velocity','max_velocity',
    'avg_spin_rate','avg_horizontal_break','avg_vertical_break','k_per_9','k_percent',
    'swinging_strike_percent','called_strike_percent','zone_percent','fastball_percent',
    'breaking_percent','offspeed_percent','total_swinging_strikes','total_called_strikes',
    'total_fastballs','total_breaking','total_offspeed','total_in_zone','pa_vs_rhb',
    'k_vs_rhb','k_percent_vs_rhb','pa_vs_lhb','k_vs_lhb','k_percent_vs_lhb',
    'platoon_split_k_pct','strikeouts_change','k_percent_change','k_per_9_change',
    'batters_faced_change','innings_pitched_change','swinging_strike_percent_change',
    'called_strike_percent_change','zone_percent_change','fastball_percent_change',
    'breaking_percent_change','offspeed_percent_change','strikeouts_lag1','strikeouts_lag2',
    'k_percent_lag1','k_percent_lag2','k_per_9_lag1','k_per_9_lag2',
    'ewma_3g_strikeouts','ewma_5g_strikeouts','ewma_10g_strikeouts',
    'ewma_3g_k_percent','ewma_5g_k_percent','ewma_10g_k_percent',
    'ewma_3g_k_per_9','ewma_5g_k_per_9','ewma_10g_k_per_9',
    'strikeouts_last2g_vs_baseline','k_percent_last2g_vs_baseline',
    'k_per_9_last2g_vs_baseline','k_trend_up_lagged','k_trend_down_lagged',
    'umpire_historical_k_per_9','pitcher_umpire_k_boost','inning','score_differential',
    'is_close_game','is_playoff','rest_days_6_more','rest_days_4_less','rest_days_5',
    'avg_velocity_imputed_median','max_velocity_imputed_median','avg_spin_rate_imputed_median',
    'avg_horizontal_break_imputed_median','avg_vertical_break_imputed_median',
    'avg_velocity_imputed_knn','avg_spin_rate_imputed_knn',
    'avg_horizontal_break_imputed_knn','avg_vertical_break_imputed_knn',
]

def calculate_vif(df, features):
    """Calculate VIF for each feature."""
    X = df[features].assign(constant=1)
    vif_data = pd.DataFrame({
        'feature': features,
        'vif': [variance_inflation_factor(X.values, i) for i in range(len(features))]
    })
    return vif_data

def prune_by_vif(df, features, threshold=10.0):
    """Iteratively remove features with VIF above threshold."""
    if variance_inflation_factor is None:
        logger.warning("statsmodels not installed; skipping VIF pruning.")
        return features
    feats = features.copy()
    while True:
        vif_df = calculate_vif(df, feats)
        max_v = vif_df['vif'].max()
        if max_v <= threshold:
            break
        drop = vif_df.sort_values('vif', ascending=False)['feature'].iloc[0]
        logger.info(f"Dropping '{drop}' (VIF={max_v:.2f})")
        feats.remove(drop)
    return feats

def prune_by_shap(model, X, features, threshold=0.0, sample_frac=0.1):
    """Drop low-impact features based on mean absolute SHAP values."""
    if shap is None:
        logger.warning("shap not installed; skipping SHAP pruning.")
        return features
    # Ensure we're using the same features the model was trained on
    try:
        trained_feats = list(model.feature_name())
    except Exception:
        logger.warning("Unable to retrieve model.feature_name(); skipping SHAP pruning.")
        return features
    shap_feats = [f for f in features if f in trained_feats]
    if not shap_feats:
        logger.error("No overlapping features for SHAP pruning; skipping.")
        return features
    sample = X[shap_feats].sample(frac=sample_frac, random_state=0)
    # Monkey-patch predict to disable shape check (LightGBM expects original training shape)
    try:
        orig_predict = model.predict
        def predict_ignore_shape(data, *args, **kwargs):
            kwargs['predict_disable_shape_check'] = True
            return orig_predict(data, *args, **kwargs)
        model.predict = predict_ignore_shape
        logger.info("Patched model.predict to disable shape check for SHAP.")
    except Exception:
        logger.warning("Could not patch model.predict; SHAP pruning may error on shape mismatch.")
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(sample)
    arr = np.abs(vals if not isinstance(vals, list) else vals[0])
    means = np.mean(arr, axis=0)
    dfv = pd.DataFrame({'feature': shap_feats, 'mean_abs_shap': means})
    drops = dfv[dfv['mean_abs_shap'] <= threshold]['feature'].tolist()
    if drops:
        logger.info(f"Dropping low-impact features via SHAP: {drops}")
    # Keep order, drop only flagged features present in shap_feats
    return [f for f in features if f in shap_feats and f not in drops]

def select_features(
    df, target_variable,
    exclude_cols=None,
    prune_vif=True, vif_threshold=10.0,
    prune_shap=False, shap_model=None, shap_threshold=0.0, shap_sample_frac=0.1
):
    """Select numeric features, exclude leakage, then optionally prune via SHAP and VIF."""
    if df is None or df.empty:
        logger.error("Empty input DataFrame.")
        return [], pd.DataFrame()

    exclude = set(exclude_cols or BASE_EXCLUDE_COLS)
    cols = {c for c in df.columns if c is not None}
    if target_variable in cols:
        exclude.add(target_variable)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    feats = [c for c in num_cols if c in cols and c not in exclude]

    # Inf check
    if feats:
        inf = np.isinf(df[feats]).any()
        if inf.any():
            bad = list(inf[inf].index)
            logger.warning(f"Infinite values in: {bad}")

    # SHAP prune
    if prune_shap and shap_model is not None:
        feats = prune_by_shap(shap_model, df, feats, threshold=shap_threshold, sample_frac=shap_sample_frac)
    # VIF prune
    if prune_vif:
        feats = prune_by_vif(df, feats, threshold=vif_threshold)

    if not feats:
        logger.error("No features selected after exclusions/pruning.")
        return [], pd.DataFrame()

    out_cols = [c for c in feats if c in cols]
    if target_variable in cols:
        out_cols.append(target_variable)
    subset = df[out_cols].copy()
    logger.info(f"Selected {len(feats)} features (+target).")
    return feats, subset