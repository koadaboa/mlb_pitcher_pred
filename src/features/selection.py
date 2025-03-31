# src/features/selection.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, TimeSeriesSplit
from src.data.utils import setup_logger
from config import StrikeoutModelConfig

logger = setup_logger(__name__)

def select_features_for_strikeout_model(df, method='rfecv', n_features=30, perform_comparison=True):
    """
    Select relevant features for strikeout prediction model
    
    Args:
        df (pandas.DataFrame): Complete dataset with features
        method (str): Selection method ('rfecv', 'importance', 'all')
        n_features (int): Number of features to select (for importance method)
        perform_comparison (bool): Whether to compare methods
        
    Returns:
        list: Selected features for strikeout model
    """
    # Check what columns are available
    available_columns = df.columns.tolist()
    logger.info(f"Available columns: {len(available_columns)}")
    
    # Define baseline features that should always be included
    baseline_features = [
        'last_3_games_strikeouts_avg', 
        'last_5_games_strikeouts_avg',
        'career_so_avg',
        'days_rest'
    ]
    
    # Define feature groups
    base_features = [
        'last_3_games_strikeouts_avg', 
        'last_5_games_strikeouts_avg',
        'last_3_games_velo_avg', 
        'last_5_games_velo_avg',
        'last_3_games_swinging_strike_pct_avg', 
        'last_5_games_swinging_strike_pct_avg',
        'days_rest'
    ]
    
    standard_dev_features = [
        'last_3_games_strikeouts_std', 
        'last_5_games_strikeouts_std',
        'last_3_games_velo_std', 
        'last_5_games_velo_std',
        'last_3_games_swinging_strike_pct_std', 
        'last_5_games_swinging_strike_pct_std'
    ]
    
    trend_features = [
        'trend_3_strikeouts', 
        'trend_5_strikeouts',
        'trend_3_release_speed_mean', 
        'trend_5_release_speed_mean',
        'trend_3_swinging_strike_pct', 
        'trend_5_swinging_strike_pct'
    ]
    
    momentum_features = [
        'momentum_3_strikeouts', 
        'momentum_5_strikeouts',
        'momentum_3_release_speed_mean', 
        'momentum_5_release_speed_mean',
        'momentum_3_swinging_strike_pct', 
        'momentum_5_swinging_strike_pct'
    ]
    
    entropy_features = [
        'pitch_entropy', 
        'prev_game_pitch_entropy'
    ]
    
    pitcher_baseline_features = [
        'career_so_avg',
        'career_so_std',
        'career_so_per_batter',
        'career_so_consistency',
        'prev_so_deviation',
        'so_deviation_3g_avg',
        'so_deviation_5g_avg',
        'is_home_game',
        'home_away_so_exp'
    ]
    
    matchup_features = [
        'opponent_strikeout_rate',
        'opponent_whiff_rate',
        'opponent_chase_rate',
        'opponent_zone_contact_rate',
        'opponent_k_vs_avg',
        'opponent_whiff_vs_avg',
        'matchup_advantage',
        'recency_weighted_matchup'
    ]
    
    # Combine all feature groups
    all_prediction_features = (
        base_features + 
        standard_dev_features + 
        trend_features + 
        momentum_features + 
        entropy_features + 
        pitcher_baseline_features + 
        matchup_features
    )
    
    # Add pitch mix features if available
    pitch_mix_cols = [col for col in available_columns if col.startswith('prev_game_pitch_pct_')]
    all_prediction_features.extend(pitch_mix_cols)
    
    # Add opponent-specific history features
    opponent_history_cols = [col for col in available_columns if col.startswith('so_vs_')]
    all_prediction_features.extend(opponent_history_cols)
    
    # Check which prediction features are available
    available_pred_features = [f for f in all_prediction_features if f in available_columns]
    logger.info(f"Found {len(available_pred_features)} relevant features")
    
    # If method is 'all', return all available features
    if method == 'all':
        logger.info(f"Using all {len(available_pred_features)} features")
        return available_pred_features
        
    # If we need to perform feature selection
    if method in ['rfecv', 'importance']:
        # Prepare data for feature selection
        target = 'strikeouts'
        if target not in df.columns:
            logger.error(f"Target column '{target}' not found in dataframe")
            return available_pred_features
            
        # Use only training years data for feature selection
        # to avoid leakage from test years
        train_years = StrikeoutModelConfig.DEFAULT_TRAIN_YEARS
        selection_df = df[df['season'].isin(train_years)].copy()
        
        # Prepare features and target
        X = selection_df[available_pred_features].copy()
        y = selection_df[target].copy()
        
        # Drop rows with NA values
        mask = ~X.isna().any(axis=1) & ~y.isna()
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Using {len(X)} rows for feature selection after dropping NA values")
        
        # Select features using specified method
        selected_features = []
        important_features = []
        rfecv_features = []
        
        # Always compute importance-based features for comparison
        important_features = _select_by_importance(X, y, n_features)
        
        # Always compute RFECV-based features for comparison
        rfecv_features = _select_by_rfecv(X, y)
        
        # Set the final selected features based on method
        if method == 'importance':
            selected_features = important_features
        else:  # rfecv
            selected_features = rfecv_features
            
        # Include baseline features even if not selected
        for feature in baseline_features:
            if feature in available_columns and feature not in selected_features:
                selected_features.append(feature)
                logger.info(f"Added baseline feature: {feature}")
                
        # Log feature selection comparison if requested
        if perform_comparison:
            _log_feature_selection_comparison(important_features, rfecv_features)
        
        logger.info(f"Selected {len(selected_features)} features for strikeout model")
        return selected_features
    else:
        logger.warning(f"Unknown feature selection method: {method}")
        return available_pred_features

def _select_by_importance(X, y, n_features=30):
    """
    Select features based on LightGBM feature importance
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        n_features (int): Number of features to select
        
    Returns:
        list: Selected features
    """
    logger.info("Selecting features by importance...")
    
    # Create LightGBM model
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=StrikeoutModelConfig.RANDOM_STATE
    )
    
    # Fit model
    model.fit(X, y)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top n features
    selected_features = importance.head(n_features)['feature'].tolist()
    
    # Log top 10 features
    logger.info("Top 10 features by importance:")
    for i, row in importance.head(10).iterrows():
        logger.info(f"{i+1}. {row['feature']}: {row['importance']:.6f}")
    
    return selected_features

def _select_by_rfecv(X, y):
    """
    Select features using Recursive Feature Elimination with Cross-Validation
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        
    Returns:
        list: Selected features
    """
    logger.info("Selecting features by RFECV...")
    
    # Create LightGBM model
    estimator = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=StrikeoutModelConfig.RANDOM_STATE
    )
    
    # Use TimeSeriesSplit if possible (data should be chronologically sorted)
    try:
        cv = TimeSeriesSplit(n_splits=5)
        logger.info("Using TimeSeriesSplit for RFECV")
    except:
        cv = KFold(n_splits=5, shuffle=True, random_state=StrikeoutModelConfig.RANDOM_STATE)
        logger.info("Using KFold for RFECV")
    
    # Create RFECV selector
    selector = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring='neg_mean_squared_error',
        min_features_to_select=10,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit selector
    selector.fit(X, y)
    
    # Get selected features
    selected_indices = selector.support_
    selected_features = X.columns[selected_indices].tolist()
    
    # Save the feature importance from RFECV
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': estimator.feature_importances_,
        'selected': selected_indices
    }).sort_values('importance', ascending=False)
    
    # Log RFECV results
    logger.info(f"RFECV selected {len(selected_features)} features")
    logger.info(f"Optimal number of features: {selector.n_features_}")
    
    # Log top selected features
    selected_importance = importance[importance['selected']].sort_values('importance', ascending=False)
    logger.info("Top 10 selected features by importance:")
    for i, row in selected_importance.head(10).iterrows():
        logger.info(f"{i+1}. {row['feature']}: {row['importance']:.6f}")
    
    return selected_features

def _log_feature_selection_comparison(importance_features, rfecv_features):
    """
    Log comparison between feature selection methods
    
    Args:
        importance_features (list): Features selected by importance
        rfecv_features (list): Features selected by RFECV
    """
    # Find common features
    common_features = set(importance_features).intersection(set(rfecv_features))
    
    # Find unique features
    importance_unique = set(importance_features) - set(rfecv_features)
    rfecv_unique = set(rfecv_features) - set(importance_features)
    
    # Log comparison
    logger.info("\n===== FEATURE SELECTION COMPARISON =====")
    logger.info(f"Importance method selected {len(importance_features)} features")
    logger.info(f"RFECV method selected {len(rfecv_features)} features")
    logger.info(f"Common features: {len(common_features)}")
    
    # Log agreement percentage
    if importance_features and rfecv_features:
        jaccard = len(common_features) / len(set(importance_features).union(set(rfecv_features)))
        logger.info(f"Jaccard similarity (agreement): {jaccard:.2f}")
    
    # Log unique features from each method
    logger.info(f"\nFeatures unique to Importance method ({len(importance_unique)}):")
    for feature in sorted(importance_unique):
        logger.info(f"- {feature}")
    
    logger.info(f"\nFeatures unique to RFECV method ({len(rfecv_unique)}):")
    for feature in sorted(rfecv_unique):
        logger.info(f"- {feature}")

# Add to src/features/selection.py

def select_features_with_rfecv(df, features=None):
    """
    Select optimal features using Recursive Feature Elimination with Cross-Validation
    
    Args:
        df (pandas.DataFrame): Complete dataset with features
        features (list, optional): Initial features to consider, if None all suitable features will be used
        
    Returns:
        list: Selected features for strikeout model
    """
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import KFold
    import lightgbm as lgb
    import numpy as np
    
    logger.info("Starting RFECV feature selection")
    
    # If no features provided, get all potential features
    if features is None:
        # Get base features from regular selection
        features = select_features_for_strikeout_model(df)
    
    # Make sure we have strikeouts column and enough samples
    if 'strikeouts' not in df.columns:
        logger.error("No 'strikeouts' column found in DataFrame")
        return features
    
    if len(df) < 500:  # Arbitrary threshold for RFECV
        logger.warning(f"Not enough samples ({len(df)}) for reliable RFECV, using standard selection")
        return features
    
    # Prepare data - drop NA rows
    X = df[features].copy().fillna(0)
    y = df['strikeouts'].copy()
    
    valid_rows = ~y.isna()
    X = X[valid_rows]
    y = y[valid_rows]
    
    # Define a custom scorer for within 1 strikeout
    from sklearn.metrics import make_scorer
    
    def within_1_strikeout(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred) <= 1) * 100
    
    within_1_scorer = make_scorer(within_1_strikeout, greater_is_better=True)
    
    # Create a lightweight LightGBM model for feature selection
    lgb_model = lgb.LGBMRegressor(
        n_estimators=50,  # Reduced for speed
        learning_rate=0.1,
        max_depth=5,
        random_state=StrikeoutModelConfig.RANDOM_STATE
    )
    
    # Create cross-validation folds
    cv = KFold(n_splits=5, shuffle=True, random_state=StrikeoutModelConfig.RANDOM_STATE)
    
    try:
        # Initialize RFECV - using smaller step to speed up process
        rfecv = RFECV(
            estimator=lgb_model,
            step=0.1,  # Remove 10% of features at each step
            min_features_to_select=10,  # Don't go below 10 features
            cv=cv,
            scoring=within_1_scorer,
            n_jobs=-1,  # Use all available cores
            verbose=1
        )

        feature_names = X.columns.tolist()
        
        # Fit RFECV
        rfecv.fit(X, y)
        
        # Get selected features
        selected_features = [feature_names[i] for i in range(len(feature_names)) if rfecv.support_[i]]

        # Add visual separators here
        logger.info("\n" + "="*50)
        logger.info("FEATURE SELECTION RESULTS")
        logger.info("="*50)
        
        # Log results
        logger.info(f"RFECV selected {len(selected_features)} out of {len(features)} features")
        logger.info(f"Optimal number of features: {rfecv.n_features_}")

        # Log ALL selected features 
        logger.info("ALL SELECTED FEATURES:")
        for feature in selected_features:
            logger.info(f"  - {feature}")
        
        # Get feature ranking
        feature_ranking = [(features[i], r) for i, r in enumerate(rfecv.ranking_)]
        feature_ranking.sort(key=lambda x: x[1])  # Sort by rank
        
        # Log top 10 features by ranking
        logger.info("TOP 10 FEATURES BY RFECV RANKING:")
        for feature, rank in feature_ranking[:10]:
            logger.info(f"  {feature} (rank: {rank})")
        
        # If too few features were selected, add more from the original set
        if len(selected_features) < 10:
            logger.warning(f"Too few features selected ({len(selected_features)}), adding more")
            
            # Get feature ranking
            feature_ranking = [(i, r) for i, r in enumerate(rfecv.ranking_)]
            feature_ranking.sort(key=lambda x: x[1])  # Sort by rank
            
            # Add features until we have at least 10
            additional_indices = [i for i, r in feature_ranking if i not in rfecv.support_][:10-len(selected_features)]
            additional_features = [features[i] for i in additional_indices]
            
            selected_features.extend(additional_features)
            logger.info(f"Added {len(additional_features)} features, now have {len(selected_features)}")
        
        return selected_features
        
    except Exception as e:
        logger.error(f"RFECV failed: {e}")
        logger.info("Falling back to standard feature selection")
        return features

def select_features_for_model(df, method='rfecv'):
    """
    Select features for strikeout prediction model with multiple methods
    
    Args:
        df (pandas.DataFrame): Complete dataset with features
        method (str): Selection method ('manual', 'rfecv', or 'both')
        
    Returns:
        list: Selected features for strikeout model
    """
    if method == 'manual':
        logger.info("Using manual feature selection")
        return select_features_for_strikeout_model(df)
    
    elif method == 'rfecv':
        logger.info("Using RFECV feature selection")
        # First get base features from manual selection
        base_features = select_features_for_strikeout_model(df)
        # Then refine with RFECV
        return select_features_with_rfecv(df, features=base_features)
    
    elif method == 'both':
        logger.info("Using combined feature selection approach")
        # Get both sets of features
        manual_features = select_features_for_strikeout_model(df)
        rfecv_features = select_features_with_rfecv(df, features=manual_features)
        
        # Combine and deduplicate
        all_features = list(set(manual_features + rfecv_features))
        logger.info(f"Combined selection: {len(all_features)} features")
        return all_features
    
    else:
        logger.warning(f"Unknown feature selection method: {method}, using manual")
        return select_features_for_strikeout_model(df)