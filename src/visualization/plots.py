# Visualization functions for pitcher performance analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data.utils import setup_logger, ensure_dir

logger = setup_logger(__name__)

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

def create_features_importance_plot(model, feature_names, viz_dir):
    """
    Create feature importance visualization for a trained model
    
    Args:
        model: Trained ML model with feature_importances_ attribute
        feature_names (list): List of feature names
        viz_dir (pathlib.Path): Directory to save visualization
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
    plt.title('Top Features for Strikeout Prediction')
    plt.tight_layout()
    plt.savefig(viz_dir / 'strikeout_feature_importance.png')
    plt.close()
    
    logger.info("Created feature importance visualization for strikeout model")
    return True

def create_predictions_vs_actual_plot(y_true, y_pred, viz_dir):
    """
    Create scatter plot of predicted vs actual values
    
    Args:
        y_true (array-like): Actual values
        y_pred (array-like): Predicted values
        viz_dir (pathlib.Path): Directory to save visualization
    """
    plt.figure(figsize=(10, 8))
    
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual Strikeouts')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(viz_dir / 'strikeout_predictions.png')
    plt.close()
    
    logger.info("Created predictions vs actual plot for strikeout model")
    return True