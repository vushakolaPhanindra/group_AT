"""
Utility functions for Credit Score Intelligence project.

This module provides common utility functions used across the ML and SHAP pipelines,
including model loading, directory creation, and project path management.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """
    Get the project root directory dynamically.
    
    Returns:
        Path object pointing to the project root directory
    """
    # Get the directory of this file (src/)
    current_file = Path(__file__).resolve()
    # Go up one level to get the project root
    project_root = current_file.parent.parent
    return project_root

def create_output_directories() -> None:
    """
    Create necessary output directories if they don't exist.
    
    Creates the following directory structure:
    - outputs/
    - outputs/plots/
    - outputs/shap_summaries/
    - outputs/rationales/
    - models/
    """
    project_root = get_project_root()
    
    directories = [
        "outputs",
        "outputs/plots",
        "outputs/shap_summaries", 
        "outputs/rationales",
        "models"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        try:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Directory created/verified: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {str(e)}")
            raise

def load_model(model_path: str) -> Optional[Any]:
    """
    Safely load a machine learning model from a file.
    
    Supports both .joblib and .pkl file formats. Tries joblib first (preferred for scikit-learn),
    then falls back to pickle if joblib is not available or fails.
    
    Args:
        model_path: Path to the model file (.joblib or .pkl)
        
    Returns:
        Loaded model object or None if loading fails
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    # Convert to absolute path if relative
    if not os.path.isabs(model_path):
        project_root = get_project_root()
        model_path = project_root / model_path
        model_path = str(model_path)
    
    # Check if file exists
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return None
    
    try:
        # Try joblib first (preferred for scikit-learn models)
        try:
            import joblib
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully using joblib: {model_path}")
            return model
        except ImportError:
            logger.warning("joblib not available, trying pickle")
        except Exception as e:
            logger.warning(f"Failed to load with joblib: {str(e)}, trying pickle")
        
        # Fallback to pickle
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded successfully using pickle: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model with pickle: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error loading model from {model_path}: {str(e)}")
        return None

def save_model(model: Any, model_path: str) -> bool:
    """
    Save a machine learning model to a file.
    
    Args:
        model: The model object to save
        model_path: Path where to save the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert to absolute path if relative
        if not os.path.isabs(model_path):
            project_root = get_project_root()
            model_path = project_root / model_path
            model_path = str(model_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Try joblib first
        try:
            import joblib
            joblib.dump(model, model_path)
            logger.info(f"Model saved successfully using joblib: {model_path}")
            return True
        except ImportError:
            logger.warning("joblib not available, using pickle")
        except Exception as e:
            logger.warning(f"Failed to save with joblib: {str(e)}, using pickle")
        
        # Fallback to pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully using pickle: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save model to {model_path}: {str(e)}")
        return False

def ensure_file_extension(file_path: str, extension: str) -> str:
    """
    Ensure a file path has the specified extension.
    
    Args:
        file_path: The file path to check
        extension: The desired extension (with or without leading dot)
        
    Returns:
        File path with the correct extension
    """
    if not extension.startswith('.'):
        extension = '.' + extension
    
    if not file_path.endswith(extension):
        file_path += extension
    
    return file_path

def get_timestamp() -> str:
    """
    Get a timestamp string for file naming.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def safe_filename(filename: str) -> str:
    """
    Convert a string to a safe filename by removing/replacing invalid characters.
    
    Args:
        filename: The original filename
        
    Returns:
        Safe filename string
    """
    import re
    # Replace invalid characters with underscores
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    return safe_name

def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('credit_score_intelligence.log')
        ]
    )

def validate_model_path(model_path: str) -> bool:
    """
    Validate that a model path exists and is readable.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Convert to absolute path if relative
        if not os.path.isabs(model_path):
            project_root = get_project_root()
            model_path = project_root / model_path
            model_path = str(model_path)
        
        return os.path.exists(model_path) and os.path.isfile(model_path)
    except Exception:
        return False

def get_model_info(model_path: str) -> dict:
    """
    Get information about a model file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with model file information
    """
    try:
        # Convert to absolute path if relative
        if not os.path.isabs(model_path):
            project_root = get_project_root()
            model_path = project_root / model_path
            model_path = str(model_path)
        
        if not os.path.exists(model_path):
            return {"exists": False, "error": "File not found"}
        
        stat = os.stat(model_path)
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": stat.st_mtime,
            "extension": os.path.splitext(model_path)[1]
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}

# Initialize output directories when module is imported
if __name__ != "__main__":
    try:
        create_output_directories()
    except Exception as e:
        logger.warning(f"Failed to create output directories: {str(e)}")