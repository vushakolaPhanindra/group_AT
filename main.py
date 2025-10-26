"""
Main entry point for credit score intelligence application.

This module orchestrates the entire pipeline from data preprocessing
to model training, explanation, and serving.
"""

import sys
import os
import logging
import pandas as pd
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from preprocess import preprocess_pipeline, load_data, clean_data, encode_features, split_data
from train_model import train_model_pipeline, load_trained_model
from explain_model import explain_model_pipeline, compute_shap_values, plot_shap_summary, plot_shap_waterfall

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_preprocessing():
    """
    Test the preprocessing pipeline to verify functionality.
    """
    logger.info("Testing preprocessing pipeline...")
    
    try:
        # Test the complete preprocessing pipeline
        X_train, X_test, y_train, y_test, label_encoders = preprocess_pipeline()
        
        # Print basic information about the processed data
        logger.info("=" * 50)
        logger.info("PREPROCESSING TEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Training target shape: {y_train.shape}")
        logger.info(f"Test target shape: {y_test.shape}")
        
        # Print column information
        logger.info(f"\nFeature columns ({len(X_train.columns)}):")
        for i, col in enumerate(X_train.columns, 1):
            logger.info(f"  {i:2d}. {col}")
        
        # Print target distribution
        logger.info(f"\nTarget distribution:")
        logger.info(f"  Training: {y_train.value_counts().to_dict()}")
        logger.info(f"  Test: {y_test.value_counts().to_dict()}")
        
        # Print label encoders information
        logger.info(f"\nLabel encoders created for {len(label_encoders)} columns:")
        for col, encoder in label_encoders.items():
            if hasattr(encoder, 'classes_'):
                logger.info(f"  {col}: {list(encoder.classes_)}")
        
        # Check if processed data file was created
        processed_file = "data/processed_credit.csv"
        if os.path.exists(processed_file):
            logger.info(f"\n✅ Processed data saved to: {processed_file}")
        else:
            logger.warning(f"⚠️  Processed data file not found: {processed_file}")
        
        logger.info("\n✅ Preprocessing test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Preprocessing test failed: {str(e)}")
        return False


def test_model_training():
    """
    Test the model training pipeline.
    """
    logger.info("Testing model training pipeline...")
    
    try:
        # Load processed data
        processed_data = pd.read_csv("data/processed_credit.csv")
        logger.info(f"Loaded processed data with shape: {processed_data.shape}")
        
        # Split data
        X = processed_data.drop(columns=['Credit_Score'])
        y = processed_data['Credit_Score']
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define class names
        class_names = ['Poor', 'Standard', 'Good']
        
        # Train model (without hyperparameter tuning for speed)
        model, results = train_model_pipeline(
            X_train, y_train, X_test, y_test, 
            class_names=class_names, 
            perform_tuning=False
        )
        
        logger.info("✅ Model training test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model training test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_explanation():
    """
    Test the model explanation pipeline.
    """
    logger.info("Testing model explanation pipeline...")
    
    try:
        # Check if model exists
        model_path = "models/credit_model.pkl"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Run explanation pipeline
        class_names = ['Poor', 'Standard', 'Good']
        results = explain_model_pipeline(
            model_path=model_path,
            data_path="data/processed_credit.csv",
            class_names=class_names
        )
        
        # Print results summary
        logger.info("=" * 60)
        logger.info("MODEL EXPLANATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"SHAP values shape: {results['shap_values_shape']}")
        logger.info(f"Expected value: {results['expected_value']}")
        logger.info(f"Number of features: {len(results['feature_names'])}")
        logger.info(f"Class names: {results['class_names']}")
        
        logger.info("\nPlots created:")
        for plot in results['plots_created']:
            if os.path.exists(plot):
                logger.info(f"  ✅ {plot}")
            else:
                logger.warning(f"  ❌ {plot} (not found)")
        
        logger.info(f"\nJSON exported: {results['json_exported']}")
        if os.path.exists(results['json_exported']):
            logger.info(f"  ✅ {results['json_exported']}")
        else:
            logger.warning(f"  ❌ {results['json_exported']} (not found)")
        
        logger.info("✅ Model explanation test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model explanation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to run the credit score intelligence pipeline.
    """
    logger.info("Starting Credit Score Intelligence Application")
    
    # Test preprocessing
    if test_preprocessing():
        logger.info("Preprocessing pipeline is working correctly!")
    else:
        logger.error("Preprocessing pipeline failed!")
        return 1
    
    # Test model training
    if test_model_training():
        logger.info("Model training pipeline is working correctly!")
    else:
        logger.error("Model training pipeline failed!")
        return 1
    
    # Test model explanation
    if test_model_explanation():
        logger.info("Model explanation pipeline is working correctly!")
    else:
        logger.error("Model explanation pipeline failed!")
        return 1
    
    # TODO: Add rationale generation pipeline
    # TODO: Add API serving
    
    logger.info("Application completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)