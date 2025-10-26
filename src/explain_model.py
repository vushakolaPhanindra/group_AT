"""
Model explanation module for credit score intelligence.

This module provides explainable AI capabilities using SHAP values
and other interpretability techniques to understand model decisions.
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP imports
import shap
from shap import TreeExplainer, summary_plot, waterfall_plot

try:
    from src.utils import load_model, create_output_directories
except ImportError:
    from utils import load_model, create_output_directories

# Set up logging
logger = logging.getLogger(__name__)


def compute_shap_values(model: Any, X: pd.DataFrame, max_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SHAP values for the given model and data.
    
    Args:
        model: Trained model (must be tree-based for TreeExplainer)
        X: Feature matrix
        max_samples: Maximum number of samples to compute SHAP values for
        
    Returns:
        Tuple of (shap_values, expected_value)
    """
    logger.info("Computing SHAP values...")
    
    # Limit samples for performance if dataset is large
    if len(X) > max_samples:
        logger.info(f"Sampling {max_samples} instances from {len(X)} total samples")
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X
    
    # Initialize SHAP TreeExplainer
    explainer = TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value
    
    # Convert to numpy array if it's a list
    if isinstance(shap_values, list):
        shap_values = np.array(shap_values)
    
    logger.info(f"SHAP values computed for {len(X_sample)} samples")
    logger.info(f"Expected value: {expected_value}")
    
    return shap_values, expected_value


def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame, 
                     class_names: List[str] = None,
                     save_path: str = "outputs/plots/shap_summary.png") -> None:
    """
    Create and save SHAP summary plot.
    
    Args:
        shap_values: SHAP values array
        X: Feature matrix used for SHAP computation
        class_names: List of class names for multi-class problems
        save_path: Path to save the plot
    """
    logger.info("Creating SHAP summary plot...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    
    if len(shap_values.shape) == 3:  # Multi-class case
        # For multi-class, we'll plot the first class (index 0)
        # SHAP values shape is (n_samples, n_features, n_classes)
        summary_plot(shap_values[:, :, 0], X, show=False, max_display=15)
    else:  # Binary case
        summary_plot(shap_values, X, show=False, max_display=15)
    
    plt.title('SHAP Summary Plot - Feature Importance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"SHAP summary plot saved to: {save_path}")


def plot_shap_waterfall(shap_values: np.ndarray, X: pd.DataFrame, 
                       sample_idx: int = 0,
                       class_names: List[str] = None,
                       save_path: str = "outputs/plots/sample_explain.png") -> None:
    """
    Create and save SHAP waterfall plot for a single sample.
    
    Args:
        shap_values: SHAP values array
        X: Feature matrix used for SHAP computation
        sample_idx: Index of the sample to explain
        class_names: List of class names for multi-class problems
        save_path: Path to save the plot
    """
    logger.info(f"Creating SHAP waterfall plot for sample {sample_idx}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get SHAP values for the specific sample
    if len(shap_values.shape) == 3:  # Multi-class case
        sample_shap_values = shap_values[sample_idx, :, 0]  # Use first class
    else:  # Binary case
        sample_shap_values = shap_values[sample_idx]
    
    # Get feature values for the sample
    sample_features = X.iloc[sample_idx]
    
    # Create waterfall plot
    plt.figure(figsize=(12, 8))
    
    # Create a simple bar plot showing SHAP values
    feature_names = X.columns.tolist()
    shap_vals = sample_shap_values
    
    # Sort by absolute SHAP value for better visualization
    sorted_indices = np.argsort(np.abs(shap_vals))[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices[:10]]  # Top 10 features
    sorted_shap_vals = [shap_vals[i] for i in sorted_indices[:10]]
    
    # Create horizontal bar plot
    colors = ['red' if val < 0 else 'blue' for val in sorted_shap_vals]
    bars = plt.barh(range(len(sorted_features)), sorted_shap_vals, color=colors, alpha=0.7)
    
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('SHAP Value', fontsize=12)
    plt.title(f'SHAP Values for Sample {sample_idx} - Top 10 Features', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_shap_vals)):
        plt.text(val + (0.01 if val >= 0 else -0.01), i, f'{val:.3f}', 
                va='center', ha='left' if val >= 0 else 'right', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"SHAP waterfall plot saved to: {save_path}")


def export_shap_values_json(shap_values: np.ndarray, X: pd.DataFrame, 
                           sample_idx: int = 0,
                           save_path: str = "outputs/shap_summaries/sample_shap.json") -> None:
    """
    Export SHAP values for a specific sample as JSON.
    
    Args:
        shap_values: SHAP values array
        X: Feature matrix used for SHAP computation
        sample_idx: Index of the sample to export
        save_path: Path to save the JSON file
    """
    logger.info(f"Exporting SHAP values for sample {sample_idx}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get SHAP values for the specific sample
    if len(shap_values.shape) == 3:  # Multi-class case
        sample_shap_values = shap_values[sample_idx, :, 0]  # Use first class
    else:  # Binary case
        sample_shap_values = shap_values[sample_idx]
    
    # Get feature values for the sample
    sample_features = X.iloc[sample_idx]
    
    # Create data structure for JSON export
    shap_data = {
        "sample_index": int(sample_idx),
        "feature_values": sample_features.to_dict(),
        "shap_values": sample_shap_values.tolist(),
        "feature_names": X.columns.tolist(),
        "total_shap_value": float(np.sum(sample_shap_values)),
        "max_positive_contributor": {
            "feature": X.columns[np.argmax(sample_shap_values)],
            "value": float(np.max(sample_shap_values))
        },
        "max_negative_contributor": {
            "feature": X.columns[np.argmin(sample_shap_values)],
            "value": float(np.min(sample_shap_values))
        }
    }
    
    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(shap_data, f, indent=2, default=str)
    
    logger.info(f"SHAP values exported to: {save_path}")


def create_feature_importance_plot(shap_values: np.ndarray, feature_names: List[str],
                                  save_path: str = "outputs/plots/feature_importance_shap.png") -> None:
    """
    Create a feature importance plot based on mean absolute SHAP values.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        save_path: Path to save the plot
    """
    logger.info("Creating feature importance plot from SHAP values...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Calculate mean absolute SHAP values
    if len(shap_values.shape) == 3:  # Multi-class case
        mean_abs_shap = np.mean(np.abs(shap_values[:, :, 0]), axis=0)
    else:  # Binary case
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Create DataFrame for easier handling
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=True)
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(importance_df)), importance_df['importance'], 
                    color='skyblue', alpha=0.7)
    
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.title('Feature Importance (Mean Absolute SHAP Values)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
        plt.text(val + 0.001, i, f'{val:.3f}', va='center', ha='left', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance plot saved to: {save_path}")


def explain_model_pipeline(model_path: str = "models/credit_model.pkl",
                          data_path: str = "data/processed_credit.csv",
                          class_names: List[str] = None) -> Dict[str, Any]:
    """
    Complete model explanation pipeline using SHAP.
    
    Args:
        model_path: Path to the trained model
        data_path: Path to the processed data
        class_names: List of class names for visualization
        
    Returns:
        Dictionary containing explanation results
    """
    logger.info("Starting model explanation pipeline...")
    
    # Create output directories
    create_output_directories()
    
    # Set default class names
    if class_names is None:
        class_names = ['Poor', 'Standard', 'Good']
    
    # Load model
    model = load_model(model_path)
    logger.info("Model loaded successfully")
    
    # Load data
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Credit_Score'])
    logger.info(f"Data loaded with shape: {X.shape}")
    
    # Compute SHAP values
    shap_values, expected_value = compute_shap_values(model, X)
    
    # Get the sampled data used for SHAP computation
    if len(X) > 100:
        X_sample = X.sample(n=100, random_state=42)
    else:
        X_sample = X
    
    # Create visualizations
    plot_shap_summary(shap_values, X_sample, class_names)
    plot_shap_waterfall(shap_values, X_sample, sample_idx=0, class_names=class_names)
    create_feature_importance_plot(shap_values, X.columns.tolist())
    
    # Export SHAP values as JSON
    export_shap_values_json(shap_values, X_sample, sample_idx=0)
    
    # Prepare results
    results = {
        'shap_values_shape': shap_values.shape,
        'expected_value': expected_value,
        'feature_names': X.columns.tolist(),
        'class_names': class_names,
        'plots_created': [
            'outputs/plots/shap_summary.png',
            'outputs/plots/sample_explain.png',
            'outputs/plots/feature_importance_shap.png'
        ],
        'json_exported': 'outputs/shap_summaries/sample_shap.json'
    }
    
    logger.info("Model explanation pipeline completed successfully!")
    return results