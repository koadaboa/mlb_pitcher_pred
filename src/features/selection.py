# src/features/selection.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, TimeSeriesSplit
from src.data.utils import setup_logger
from config import StrikeoutModelConfig

logger = setup_logger(__name__)

# In src/features/selection.py

def select_features(df, method='rfecv', n_features=30):
    """
    Select relevant features for strikeout prediction model using RFECV by default
    
    Args:
        df (pandas.DataFrame): Complete dataset with features
        method (str): Selection method ('rfecv', 'importance', 'all'), defaults to 'rfecv'
        n_features (int): Number of features to select for importance method
        
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
    
    # Define all potential feature groups
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
        
    # Target variable
    target = 'strikeouts'
    if target not in df.columns:
        logger.error(f"Target column '{target}' not found in dataframe")
        return available_pred_features
        
    # Use only training years data for feature selection
    from config import StrikeoutModelConfig
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
    
    # Select features based on method
    selected_features = []
    
    if method == 'importance':
        # Feature selection based on importance
        import lightgbm as lgb
        from config import StrikeoutModelConfig
        
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
    else:
        # Default to RFECV method
        from sklearn.feature_selection import RFECV
        from sklearn.model_selection import KFold
        import lightgbm as lgb
        import numpy as np
        from sklearn.metrics import make_scorer
        from config import StrikeoutModelConfig
        
        logger.info("Starting RFECV feature selection")
        
        # Define a custom scorer for within 1 strikeout accuracy
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
            # Initialize RFECV 
            rfecv = RFECV(
                estimator=lgb_model,
                step=0.1,  # Remove 10% of features at each step
                min_features_to_select=10,  # Don't go below 10 features
                cv=cv,
                scoring=within_1_scorer,
                n_jobs=-1,  # Use all available cores
                verbose=1
            )

            # Fit RFECV
            rfecv.fit(X, y)
            
            # Get selected features
            selected_features = [X.columns[i] for i in range(len(X.columns)) if rfecv.support_[i]]

            # Log results
            logger.info(f"RFECV selected {len(selected_features)} out of {len(X.columns)} features")
            logger.info(f"Optimal number of features: {rfecv.n_features_}")

            # If too few features were selected, add more from the original set
            if len(selected_features) < 10:
                logger.warning(f"Too few features selected ({len(selected_features)}), adding more")
                
                # Get feature ranking
                feature_ranking = [(i, r) for i, r in enumerate(rfecv.ranking_)]
                feature_ranking.sort(key=lambda x: x[1])  # Sort by rank
                
                # Add features until we have at least 10
                additional_indices = [i for i, r in feature_ranking if X.columns[i] not in selected_features][:10-len(selected_features)]
                additional_features = [X.columns[i] for i in additional_indices]
                
                selected_features.extend(additional_features)
                logger.info(f"Added {len(additional_features)} features, now have {len(selected_features)}")
        
        except Exception as e:
            logger.error(f"RFECV failed: {e}")
            # Fall back to importance-based selection if RFECV fails
            logger.info("Falling back to importance-based feature selection")
            import lightgbm as lgb
            
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
            
            # Select top features
            selected_features = importance.head(n_features)['feature'].tolist()
    
    # Include baseline features even if not selected
    for feature in baseline_features:
        if feature in available_columns and feature not in selected_features:
            selected_features.append(feature)
            logger.info(f"Added baseline feature: {feature}")
    
    logger.info(f"Selected {len(selected_features)} features for strikeout model")
    return selected_features