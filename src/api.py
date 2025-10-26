"""
FastAPI application for Credit Score Intelligence.

This module provides REST API endpoints for credit score prediction
and explanation using SHAP values and LLM-generated rationales.
"""

import os
import logging
import json
import tempfile
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import our modules
try:
    # Try importing from src directory (when running from project root)
    from src.train_model import load_trained_model, predict_credit_score
    from src.explain_model import compute_shap_values, plot_shap_waterfall, export_shap_values_json
    from src.generate_rationale import generate_rationale_pipeline, load_shap_data
    from src.preprocess import encode_features
    from src.utils import create_output_directories
except ImportError:
    # Fallback to direct imports (when running from src directory)
    from train_model import load_trained_model, predict_credit_score
    from explain_model import compute_shap_values, plot_shap_waterfall, export_shap_values_json
    from generate_rationale import generate_rationale_pipeline, load_shap_data
    from preprocess import encode_features
    from utils import create_output_directories

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Score Intelligence API",
    description="AI-powered credit score prediction and explanation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and encoders
model = None
label_encoders = {}
feature_names = []

# Pydantic models for request/response
class CreditScoreRequest(BaseModel):
    """Request model for credit score prediction."""
    Age: int = Field(..., ge=18, le=100, description="Age of the applicant")
    Gender: str = Field(..., description="Gender (Male, Female, Other)")
    Income: float = Field(..., ge=0, description="Annual income")
    Education: str = Field(..., description="Education level (High School, Bachelor, Master, PhD)")
    Marital_Status: str = Field(..., description="Marital status (Single, Married, Divorced)")
    Number_of_Dependents: int = Field(..., ge=0, le=10, description="Number of dependents")
    Credit_History_Length: int = Field(..., ge=0, le=50, description="Credit history length in years")
    Number_of_Credit_Cards: int = Field(..., ge=0, le=20, description="Number of credit cards")
    Number_of_Bank_Accounts: int = Field(..., ge=0, le=10, description="Number of bank accounts")
    Interest_Rate: float = Field(..., ge=0, le=50, description="Interest rate percentage")
    Number_of_Loans: int = Field(..., ge=0, le=10, description="Number of active loans")
    Delay_from_due_date: int = Field(..., ge=0, le=365, description="Days delayed from due date")
    Outstanding_Debt: float = Field(..., ge=0, description="Outstanding debt amount")
    Credit_Utilization_Ratio: float = Field(..., ge=0, le=1, description="Credit utilization ratio")

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    category: str = Field(..., description="Predicted credit score category")
    confidence: float = Field(..., description="Prediction confidence score")
    feature_importance: Dict[str, float] = Field(..., description="Top feature importance scores")

class ExplanationResponse(BaseModel):
    """Response model for explanation endpoint."""
    category: str = Field(..., description="Predicted credit score category")
    rationale: str = Field(..., description="LLM-generated explanation")
    shap_plot: str = Field(..., description="Path to SHAP waterfall plot")
    shap_data: Dict[str, Any] = Field(..., description="SHAP analysis data")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the model and encoders on startup."""
    global model, label_encoders, feature_names
    
    logger.info("Starting Credit Score Intelligence API...")
    
    try:
        # Create output directories
        create_output_directories()
        
        # Load trained model
        model_path = "models/credit_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = load_trained_model(model_path)
        logger.info("Model loaded successfully")
        
        # Load label encoders (if available)
        encoders_path = "models/label_encoders.json"
        if os.path.exists(encoders_path):
            with open(encoders_path, 'r') as f:
                label_encoders = json.load(f)
            logger.info("Label encoders loaded successfully")
        
        # Define feature names
        feature_names = [
            'Age', 'Gender', 'Income', 'Education', 'Marital_Status',
            'Number_of_Dependents', 'Credit_History_Length', 'Number_of_Credit_Cards',
            'Number_of_Bank_Accounts', 'Interest_Rate', 'Number_of_Loans',
            'Delay_from_due_date', 'Outstanding_Debt', 'Credit_Utilization_Ratio'
        ]
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

def preprocess_input_data(request_data: CreditScoreRequest) -> pd.DataFrame:
    """
    Preprocess input data for prediction.
    
    Args:
        request_data: Credit score request data
        
    Returns:
        Preprocessed DataFrame
    """
    # Convert to dictionary
    data_dict = request_data.dict()
    
    # Create DataFrame
    df = pd.DataFrame([data_dict])
    
    # Encode categorical features
    categorical_columns = ['Gender', 'Education', 'Marital_Status']
    
    for col in categorical_columns:
        if col in df.columns:
            # Use label encoders if available, otherwise use simple mapping
            if col in label_encoders:
                # This would need to be implemented based on your encoder format
                pass
            else:
                # Simple encoding for demo purposes
                if col == 'Gender':
                    df[col] = df[col].map({'Male': 1, 'Female': 0, 'Other': 2})
                elif col == 'Education':
                    df[col] = df[col].map({'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3})
                elif col == 'Marital_Status':
                    df[col] = df[col].map({'Single': 0, 'Married': 1, 'Divorced': 2})
    
    return df

def get_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """
    Get feature importance from the model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        Dictionary of feature importance scores
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            # Sort by importance and return top 5
            sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])
            return sorted_importance
        else:
            return {}
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return {}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Score Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "explain": "/explain",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_credit_score_endpoint(request: CreditScoreRequest):
    """
    Predict credit score category for given input data.
    
    Args:
        request: Credit score request data
        
    Returns:
        Prediction response with category and confidence
    """
    try:
        # Convert request to dictionary
        request_data = request.dict()
        
        # Use the new predict_credit_score function
        result = predict_credit_score(request_data)
        
        logger.info(f"Prediction completed: {result['category']} (confidence: {result['confidence']:.3f})")
        
        return PredictionResponse(
            category=result['category'],
            confidence=result['confidence'],
            feature_importance=result['feature_importance']
        )
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def explain_credit_score_endpoint(request: CreditScoreRequest, background_tasks: BackgroundTasks):
    """
    Explain credit score prediction with SHAP values and LLM rationale.
    
    Args:
        request: Credit score request data
        background_tasks: FastAPI background tasks
        
    Returns:
        Explanation response with rationale and SHAP plot
    """
    try:
        # Convert request to dictionary
        request_data = request.dict()
        
        # Use the new predict_credit_score function
        result = predict_credit_score(request_data)
        
        # Create SHAP data structure
        shap_data = {
            "feature_values": request_data,
            "shap_values": result['shap_values'],
            "total_shap_value": sum(result['shap_values'])
        }
        
        # Generate a simple plot path (placeholder)
        plot_path = f"outputs/plots/sample_explain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        logger.info(f"Explanation completed for category: {result['category']}")
        
        return ExplanationResponse(
            category=result['category'],
            rationale=result['rationale'],
            shap_plot=plot_path,
            shap_data=shap_data
        )
        
    except Exception as e:
        logger.error(f"Error in explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

# Additional utility endpoints
@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model."""
    try:
        from src.train_model import get_model_info as get_model_info_func
    except ImportError:
        from train_model import get_model_info as get_model_info_func
    return get_model_info_func()

@app.get("/feature_importance")
async def get_feature_importance_endpoint():
    """Get feature importance from the model."""
    try:
        from src.train_model import get_model_info
    except ImportError:
        from train_model import get_model_info
    model_info = get_model_info()
    if not model_info["loaded"]:
        return {"feature_importance": {}}
    
    # Return dummy feature importance for now
    return {
        "feature_importance": {
            "Income": 0.3,
            "Credit_Utilization_Ratio": 0.25,
            "Credit_History_Length": 0.2,
            "Interest_Rate": 0.15,
            "Outstanding_Debt": 0.1
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)