# src/scripts/train_models.py
import logging
import os
from pathlib import Path
import pandas as pd

from src.data.db import get_pitcher_data, export_dataset_to_csv
from src.features.selection import select_features_for_strikeout_model, select_features_for_outs_model
from src.models.train import train_strikeout_model, train_outs_model, save_model
from src.visualization.plots import create_visualizations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pitcher_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_and_save_models(train_years=(2019, 2020, 2021, 2022), test_years=(2023, 2024)):
    """Train and save the strikeout and outs prediction models"""
    # Create output directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Get the data
    logger.info("Loading pitcher data...")
    pitcher_data = get_pitcher_data()
    
    # Select features for strikeout model
    logger.info("Selecting features for strikeout model...")
    so_features = select_features_for_strikeout_model(pitcher_data)
    logger.info(f"Selected {len(so_features)} features for strikeout model: {so_features}")
    
    # Train strikeout model with time-based splitting
    logger.info(f"Training strikeout prediction model with years {train_years} for training and {test_years} for testing...")
    so_model_dict = train_strikeout_model(
        pitcher_data, 
        so_features,
        train_years=train_years,
        test_years=test_years
    )
    
    # Add training and test years to model dictionary for reference
    so_model_dict['train_years'] = train_years
    so_model_dict['test_years'] = test_years
    
    # Save strikeout model
    logger.info("Saving strikeout model...")
    save_model(so_model_dict, models_dir / "strikeout_model.pkl")
    
    # Select features for outs model
    logger.info("Selecting features for outs model...")
    outs_features = select_features_for_outs_model(pitcher_data)
    logger.info(f"Selected {len(outs_features)} features for outs model: {outs_features}")
    
    # Train outs model
    logger.info(f"Training outs prediction model with years {train_years} for training and {test_years} for testing...")
    outs_model_dict = train_outs_model(
        pitcher_data,
        outs_features,
        train_years=train_years,
        test_years=test_years
    )
    
    # Add training and test years to model dictionary
    outs_model_dict['train_years'] = train_years
    outs_model_dict['test_years'] = test_years
    
    # Save outs model
    logger.info("Saving outs model...")
    save_model(outs_model_dict, models_dir / "outs_model.pkl")
    
    logger.info("Model training complete!")
    
    return {
        'strikeout_model': so_model_dict,
        'outs_model': outs_model_dict
    }

if __name__ == "__main__":
    train_and_save_models()