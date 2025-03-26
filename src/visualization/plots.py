# Visualization functions for pitcher performance analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def setup_visualization_environment():
    """Set up the visualization environment with consistent styling"""
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({'figure.figsize': (12, 8)})
    
    # Create directory for visualizations
    viz_dir = Path("data/visualizations")
    viz_dir.mkdir(exist_ok=True, parents=True)
    
    return viz_dir

def create_strikeout_distribution_plot(df, viz_dir):
    """
    Create distribution plot of strikeouts per game
    
    Args:
        df (pandas.DataFrame): Dataset with strikeout data
        viz_dir (pathlib.Path): Directory to save visualization
    """
    if 'strikeouts' not in df.columns:
        logger.warning("No 'strikeouts' column found for distribution plot")
        return False
    
    plt.figure()
    sns.histplot(df['strikeouts'], bins=20, kde=True)
    plt.title('Distribution of Strikeouts per Game')
    plt.xlabel('Strikeouts')
    plt.ylabel('Frequency')
    plt.savefig(viz_dir / 'strikeout_distribution.png')
    plt.close()
    logger.info("Created strikeout distribution visualization")
    return True

def create_strikeout_correlations_plot(df, viz_dir, top_n=20):
    """
    Create correlation plot for strikeouts with other features
    
    Args:
        df (pandas.DataFrame): Dataset with strikeout data
        viz_dir (pathlib.Path): Directory to save visualization
        top_n (int): Number of top correlations to display
    """
    if 'strikeouts' not in df.columns:
        logger.warning("No 'strikeouts' column found for correlation plot")
        return False
    
    plt.figure(figsize=(14, 10))
    
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    try:
        strikeout_corr = df[numeric_cols].corr()['strikeouts'].sort_values(ascending=False)
        
        # Plot top N correlations (or fewer if not enough)
        corr_count = min(top_n, len(strikeout_corr) - 1)
        if corr_count > 0:
            top_corr = strikeout_corr.iloc[1:corr_count+1]  # Skip self-correlation
            sns.barplot(x=top_corr.values, y=top_corr.index)
            plt.title('Top Features Correlated with Strikeouts')
            plt.tight_layout()
            plt.savefig(viz_dir / 'strikeout_correlations.png')
            plt.close()
            logger.info("Created strikeout correlations visualization")
            return True
    except Exception as e:
        logger.error(f"Error creating strikeout correlations plot: {e}")
    
    return False

def create_velocity_strikeout_plot(df, viz_dir):
    """
    Create scatter plot of velocity vs strikeouts
    
    Args:
        df (pandas.DataFrame): Dataset with velocity and strikeout data
        viz_dir (pathlib.Path): Directory to save visualization
    """
    if not all(col in df.columns for col in ['release_speed_mean', 'strikeouts']):
        logger.warning("Missing required columns for velocity vs strikeouts plot")
        return False
    
    plt.figure()
    sns.scatterplot(x='release_speed_mean', y='strikeouts', data=df)
    plt.title('Velocity vs Strikeouts')
    plt.xlabel('Average Release Speed (mph)')
    plt.ylabel('Strikeouts')
    plt.savefig(viz_dir / 'velocity_vs_strikeouts.png')
    plt.close()
    logger.info("Created velocity vs strikeouts visualization")
    return True

def create_era_distribution_plot(df, viz_dir):
    """
    Create distribution plot of ERA
    
    Args:
        df (pandas.DataFrame): Dataset with ERA data
        viz_dir (pathlib.Path): Directory to save visualization
    """
    # Check for ERA column with different possible names
    era_col = None
    for possible_col in ['era', 'ERA', 'era_x', 'era_y']:
        if possible_col in df.columns:
            era_col = possible_col
            break
    
    if not era_col:
        logger.warning("No ERA column found for visualization")
        return False
    
    logger.info(f"Using '{era_col}' column for ERA visualizations")
    plt.figure()
    # Limit to reasonable ERA values (0-10) to avoid extreme outliers
    era_data = df[df[era_col] < 10]
    sns.histplot(era_data[era_col], bins=20, kde=True)
    plt.title('Distribution of ERA')
    plt.xlabel('ERA')
    plt.ylabel('Frequency')
    plt.savefig(viz_dir / 'era_distribution.png')
    plt.close()
    logger.info("Created ERA distribution visualization")
    return True

def create_era_correlations_plot(df, viz_dir, top_n=10):
    """
    Create correlation plot for ERA with other features
    
    Args:
        df (pandas.DataFrame): Dataset with ERA data
        viz_dir (pathlib.Path): Directory to save visualization
        top_n (int): Number of top correlations (both positive and negative) to display
    """
    # Check for ERA column with different possible names
    era_col = None
    for possible_col in ['era', 'ERA', 'era_x', 'era_y']:
        if possible_col in df.columns:
            era_col = possible_col
            break
    
    if not era_col:
        logger.warning("No ERA column found for correlation visualization")
        return False
    
    plt.figure(figsize=(14, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if era_col in numeric_cols:
        try:
            era_corr = df[numeric_cols].corr()[era_col].sort_values()
            
            # Plot top correlations (both positive and negative)
            top_count = min(top_n, len(era_corr) // 2)
            if top_count > 0:
                top_era_corr = pd.concat([era_corr.iloc[:top_count], era_corr.iloc[-top_count:]])
                sns.barplot(x=top_era_corr.values, y=top_era_corr.index)
                plt.title('Features Most Correlated with ERA')
                plt.tight_layout()
                plt.savefig(viz_dir / 'era_correlations.png')
                plt.close()
                logger.info("Created ERA correlations visualization")
                return True
        except Exception as e:
            logger.error(f"Error creating ERA correlations plot: {e}")
    else:
        logger.warning(f"Column '{era_col}' not found in numeric columns for correlation")
    
    return False

def create_pitch_mix_visualization(df, viz_dir, top_n=10):
    """
    Create pitch mix visualization for top strikeout pitchers
    
    Args:
        df (pandas.DataFrame): Dataset with pitch mix and strikeout data
        viz_dir (pathlib.Path): Directory to save visualization
        top_n (int): Number of top pitchers to display
    """
    pitch_cols = [col for col in df.columns if col.startswith('pitch_pct_')]
    if not (pitch_cols and 'player_name' in df.columns and 'strikeouts' in df.columns):
        logger.warning("Missing required columns for pitch mix visualization")
        return False
    
    plt.figure(figsize=(14, 10))
    
    try:
        # Get top N pitchers by strikeout count (or fewer if not enough data)
        top_pitcher_count = min(top_n, len(df['player_name'].unique()))
        top_k_pitchers = df.groupby('player_name')['strikeouts'].mean().sort_values(ascending=False).head(top_pitcher_count)
        
        if not top_k_pitchers.empty:
            # Create pitch mix dataset for top pitchers
            pitch_mix_data = []
            for pitcher in top_k_pitchers.index:
                pitcher_data = df[df['player_name'] == pitcher][pitch_cols].mean()
                pitcher_data['pitcher'] = pitcher
                pitch_mix_data.append(pitcher_data)
            
            if pitch_mix_data:
                pitch_mix_df = pd.DataFrame(pitch_mix_data).set_index('pitcher')
                
                # Plot pitch mix
                pitch_mix_df.plot(kind='bar', stacked=True)
                plt.title(f'Pitch Mix for Top {top_pitcher_count} Strikeout Pitchers')
                plt.ylabel('Percentage')
                plt.legend(title='Pitch Type')
                plt.tight_layout()
                plt.savefig(viz_dir / 'top_pitcher_pitch_mix.png')
                plt.close()
                logger.info("Created pitch mix visualization")
                return True
    except Exception as e:
        logger.error(f"Error creating pitch mix visualization: {e}")
    
    return False

def create_features_importance_plot(model, feature_names, viz_dir, model_type='strikeout'):
    """
    Create feature importance visualization for a trained model
    
    Args:
        model: Trained ML model with feature_importances_ attribute
        feature_names (list): List of feature names
        viz_dir (pathlib.Path): Directory to save visualization
        model_type (str): Type of model ('strikeout' or 'era')
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model doesn't have feature_importances_ attribute")
        return False
    
    plt.figure(figsize=(12, 8))
    
    # Sort features by importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features or all if fewer
    top_n = min(20, len(feature_names))
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    plt.barh(range(top_n), top_importances, align='center')
    plt.yticks(range(top_n), top_features)
    plt.xlabel('Feature Importance')
    plt.title(f'Top Features for {model_type.capitalize()} Prediction')
    plt.tight_layout()
    plt.savefig(viz_dir / f'{model_type}_feature_importance.png')
    plt.close()
    
    logger.info(f"Created feature importance visualization for {model_type} model")
    return True

def create_predictions_vs_actual_plot(y_true, y_pred, viz_dir, model_type='strikeout'):
    """
    Create scatter plot of predicted vs actual values
    
    Args:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
        viz_dir (pathlib.Path): Directory to save visualization
        model_type (str): Type of model ('strikeout' or 'era')
    """
    plt.figure(figsize=(10, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Predicted vs Actual {model_type.capitalize()}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(viz_dir / f'{model_type}_predictions.png')
    plt.close()
    
    logger.info(f"Created predictions vs actual plot for {model_type} model")
    return True

def create_visualizations(df):
    """
    Create a set of visualizations for the dataset
    
    Args:
        df (pandas.DataFrame): Dataset with pitcher performance data
    """
    logger.info("Creating visualizations...")
    
    # Set up visualization environment
    viz_dir = setup_visualization_environment()
    
    # Create basic visualizations
    created_count = 0
    
    # 1. Distribution of strikeouts per game
    if create_strikeout_distribution_plot(df, viz_dir):
        created_count += 1
    
    # 2. Correlation between metrics and strikeouts
    if create_strikeout_correlations_plot(df, viz_dir):
        created_count += 1
    
    # 3. Velocity vs Strikeouts
    if create_velocity_strikeout_plot(df, viz_dir):
        created_count += 1
    
    # 4. ERA distribution
    if create_era_distribution_plot(df, viz_dir):
        created_count += 1
    
    # 5. Correlation between metrics and ERA
    if create_era_correlations_plot(df, viz_dir):
        created_count += 1
    
    # 6. Pitch Mix visualization for top strikeout pitchers
    if create_pitch_mix_visualization(df, viz_dir):
        created_count += 1
    
    logger.info(f"Created {created_count} visualizations in {viz_dir}")
    return viz_dir