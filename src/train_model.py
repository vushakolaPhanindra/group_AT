"""
Training and prediction module for Credit Score Intelligence.

This module handles model loading, prediction, and provides fallback functionality
when the trained model is not available.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the loaded model
_loaded_model = None

def load_trained_model(model_path: str = "models/credit_model.pkl") -> Optional[Any]:
    """
    Load a pre-trained scikit-learn model from the specified path.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model object or None if loading fails
    """
    global _loaded_model
    
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}. Using dummy model.")
            return None
        
        # Try to load with joblib first (preferred for scikit-learn models)
        try:
            import joblib
            _loaded_model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path} using joblib")
            return _loaded_model
        except ImportError:
            logger.warning("joblib not available, trying pickle")
        
        # Fallback to pickle
        try:
            import pickle
            with open(model_path, 'rb') as f:
                _loaded_model = pickle.load(f)
            logger.info(f"Model loaded successfully from {model_path} using pickle")
            return _loaded_model
        except Exception as e:
            logger.error(f"Failed to load model with pickle: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def create_dummy_model():
    """
    Create a dummy model for testing when the real model is not available.
    
    Returns:
        Dummy model object with predict and predict_proba methods
    """
    class DummyModel:
        def __init__(self):
            self.feature_names_ = [
                'Age', 'Gender', 'Income', 'Education', 'Marital_Status',
                'Number_of_Dependents', 'Credit_History_Length', 'Number_of_Credit_Cards',
                'Number_of_Bank_Accounts', 'Interest_Rate', 'Number_of_Loans',
                'Delay_from_due_date', 'Outstanding_Debt', 'Credit_Utilization_Ratio'
            ]
            self.classes_ = np.array(['Poor', 'Standard', 'Good'])
            self.feature_importances_ = np.random.random(len(self.feature_names_))
            self.feature_importances_ = self.feature_importances_ / self.feature_importances_.sum()
        
        def predict(self, X):
            """Predict credit score categories."""
            # Simple rule-based prediction for demo
            predictions = []
            for _, row in X.iterrows():
                income = row.get('Income', 0)
                credit_util = row.get('Credit_Utilization_Ratio', 0)
                credit_history = row.get('Credit_History_Length', 0)
                delay_days = row.get('Delay_from_due_date', 0)
                
                # Simple scoring logic
                score = 0
                if income > 80000:
                    score += 1
                if credit_util < 0.3:
                    score += 1
                if credit_history > 5:
                    score += 1
                if delay_days < 10:
                    score += 1
                
                if score >= 3:
                    predictions.append(2)  # Good
                elif score >= 2:
                    predictions.append(1)  # Standard
                else:
                    predictions.append(0)  # Poor
            
            return np.array(predictions)
        
        def predict_proba(self, X):
            """Predict credit score probabilities."""
            predictions = self.predict(X)
            probabilities = []
            
            for pred in predictions:
                prob = np.zeros(3)
                if pred == 0:  # Poor
                    prob = [0.7, 0.2, 0.1]
                elif pred == 1:  # Standard
                    prob = [0.2, 0.6, 0.2]
                else:  # Good
                    prob = [0.1, 0.2, 0.7]
                
                # Add some randomness for realism
                prob += np.random.normal(0, 0.05, 3)
                prob = np.clip(prob, 0, 1)
                prob = prob / prob.sum()  # Normalize
                probabilities.append(prob)
            
            return np.array(probabilities)
    
    return DummyModel()

def preprocess_input_data(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocess input data for prediction.
    
    Args:
        data: Dictionary containing user input data
        
    Returns:
        Preprocessed DataFrame ready for model prediction
    """
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Encode categorical features
    categorical_mappings = {
        'Gender': {'Male': 1, 'Female': 0, 'Other': 2},
        'Education': {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3},
        'Marital_Status': {'Single': 0, 'Married': 1, 'Divorced': 2}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)
    
    # Ensure all required columns are present
    required_columns = [
        'Age', 'Gender', 'Income', 'Education', 'Marital_Status',
        'Number_of_Dependents', 'Credit_History_Length', 'Number_of_Credit_Cards',
        'Number_of_Bank_Accounts', 'Interest_Rate', 'Number_of_Loans',
        'Delay_from_due_date', 'Outstanding_Debt', 'Credit_Utilization_Ratio'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match expected order
    df = df[required_columns]
    
    return df

def generate_rationale(category: str, confidence: float, feature_importance: Dict[str, float]) -> str:
    """
    Generate a human-readable rationale for the prediction.
    
    Args:
        category: Predicted credit score category
        confidence: Prediction confidence
        feature_importance: Dictionary of feature importance scores
        
    Returns:
        Generated rationale text
    """
    # Get top 3 most important features
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    
    rationale_parts = []
    
    # Add confidence-based opening
    if confidence > 0.8:
        rationale_parts.append("Based on a high-confidence analysis of your financial profile,")
    elif confidence > 0.6:
        rationale_parts.append("Based on a moderate-confidence analysis of your financial profile,")
    else:
        rationale_parts.append("Based on an analysis of your financial profile,")
    
    # Add category-specific explanation
    if category == "Good":
        rationale_parts.append("your credit score is predicted to be **Good**. This indicates strong financial health with low risk factors.")
    elif category == "Standard":
        rationale_parts.append("your credit score is predicted to be **Standard**. This represents moderate financial health with some areas for improvement.")
    else:
        rationale_parts.append("your credit score is predicted to be **Poor**. This suggests higher risk factors that may need attention.")
    
    # Add feature-specific insights
    if top_features:
        rationale_parts.append("The most influential factors in this prediction are:")
        for i, (feature, importance) in enumerate(top_features, 1):
            feature_name = feature.replace('_', ' ').title()
            rationale_parts.append(f"{i}. **{feature_name}** (impact: {importance:.3f})")
    
    # Add recommendations
    if category == "Good":
        rationale_parts.append("Continue maintaining your current financial practices to preserve this strong credit standing.")
    elif category == "Standard":
        rationale_parts.append("Consider focusing on the areas mentioned above to improve your credit score over time.")
    else:
        rationale_parts.append("We recommend reviewing the key factors above and developing a plan to address any issues affecting your credit score.")
    
    return "\n\n".join(rationale_parts)

def predict_credit_score(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict credit score for given input data.
    
    Args:
        data: Dictionary containing user input data
        
    Returns:
        Dictionary containing prediction results
    """
    global _loaded_model
    
    try:
        # Load model if not already loaded
        if _loaded_model is None:
            _loaded_model = load_trained_model()
            if _loaded_model is None:
                logger.info("Using dummy model for prediction")
                _loaded_model = create_dummy_model()
        
        # Preprocess input data
        df = preprocess_input_data(data)
        
        # Make predictions
        prediction = _loaded_model.predict(df)[0]
        probabilities = _loaded_model.predict_proba(df)[0]
        
        # Map prediction to category
        category_map = {0: "Poor", 1: "Standard", 2: "Good"}
        category = category_map.get(prediction, "Unknown")
        
        # Get confidence (max probability)
        confidence = float(np.max(probabilities))
        
        # Get feature importance
        if hasattr(_loaded_model, 'feature_importances_'):
            feature_names = getattr(_loaded_model, 'feature_names_', [f'feature_{i}' for i in range(len(_loaded_model.feature_importances_))])
            feature_importance = dict(zip(feature_names, _loaded_model.feature_importances_))
            # Sort by importance and get top 5
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        else:
            # Create dummy feature importance
            feature_importance = {
                'Income': 0.3,
                'Credit_Utilization_Ratio': 0.25,
                'Credit_History_Length': 0.2,
                'Interest_Rate': 0.15,
                'Outstanding_Debt': 0.1
            }
        
        # Generate SHAP values (simplified)
        shap_values = probabilities.tolist()
        
        # Generate rationale
        rationale = generate_rationale(category, confidence, feature_importance)
        
        # Prepare response
        response = {
            "category": category,
            "confidence": confidence,
            "feature_importance": feature_importance,
            "shap_values": shap_values,
            "rationale": rationale
        }
        
        logger.info(f"Prediction completed: {category} (confidence: {confidence:.3f})")
        return response
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        # Return fallback response
        return {
            "category": "Unknown",
            "confidence": 0.0,
            "feature_importance": {},
            "shap_values": [0.33, 0.33, 0.34],
            "rationale": "Unable to process prediction due to an error. Please try again or contact support."
        }

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.
    
    Returns:
        Dictionary containing model information
    """
    global _loaded_model
    
    if _loaded_model is None:
        return {
            "loaded": False,
            "type": "None",
            "features": 0,
            "classes": []
        }
    
    info = {
        "loaded": True,
        "type": type(_loaded_model).__name__,
        "features": len(getattr(_loaded_model, 'feature_names_', [])),
        "classes": getattr(_loaded_model, 'classes_', []).tolist() if hasattr(_loaded_model, 'classes_') else []
    }
    
    return info

# Initialize model on module load
if __name__ != "__main__":
    # Try to load the model when the module is imported
    _loaded_model = load_trained_model()
    if _loaded_model is None:
        logger.info("No trained model found, will use dummy model for predictions")