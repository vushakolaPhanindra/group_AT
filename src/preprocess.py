"""
Data preprocessing module for credit score intelligence.

This module handles data cleaning, feature engineering, and preparation
of credit score datasets for machine learning model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_path: str = "data/credit_score.csv") -> pd.DataFrame:
    """
    Load credit score dataset from file.
    
    Args:
        data_path: Path to the credit score dataset file
        
    Returns:
        pd.DataFrame: Loaded credit score dataset
        
    Raises:
        FileNotFoundError: If the data file is not found
    """
    try:
        if not os.path.exists(data_path):
            # If file doesn't exist, create a sample dataset for demonstration
            logger.warning(f"Data file {data_path} not found. Creating sample dataset...")
            return _create_sample_dataset()
        
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def _create_sample_dataset() -> pd.DataFrame:
    """
    Create a sample credit score dataset for demonstration purposes.
    
    Returns:
        pd.DataFrame: Sample credit score dataset
    """
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(18, 80, n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n_samples),
        'Income': np.random.normal(50000, 20000, n_samples),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'Number_of_Dependents': np.random.randint(0, 6, n_samples),
        'Credit_History_Length': np.random.randint(1, 30, n_samples),
        'Number_of_Credit_Cards': np.random.randint(0, 10, n_samples),
        'Number_of_Bank_Accounts': np.random.randint(1, 5, n_samples),
        'Interest_Rate': np.random.uniform(3.0, 25.0, n_samples),
        'Number_of_Loans': np.random.randint(0, 5, n_samples),
        'Delay_from_due_date': np.random.randint(0, 30, n_samples),
        'Outstanding_Debt': np.random.uniform(0, 100000, n_samples),
        'Credit_Utilization_Ratio': np.random.uniform(0, 1, n_samples),
        'Credit_Score': np.random.choice(['Poor', 'Standard', 'Good'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values to simulate real data
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'Income'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[missing_indices, 'Education'] = np.nan
    
    logger.info("Created sample dataset for demonstration")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the credit score dataset by handling missing values and outliers.
    
    Args:
        df: Raw credit score dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    logger.info("Starting data cleaning...")
    
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Handle missing values
    # For numeric columns, use median imputation
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    imputer_numeric = SimpleImputer(strategy='median')
    df_clean[numeric_columns] = imputer_numeric.fit_transform(df_clean[numeric_columns])
    
    # For categorical columns, use mode imputation
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    df_clean[categorical_columns] = imputer_categorical.fit_transform(df_clean[categorical_columns])
    
    # Handle outliers using IQR method for numeric columns
    for col in numeric_columns:
        if col != 'Credit_Score':  # Don't apply outlier removal to target
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    # Ensure Income is positive
    df_clean = df_clean[df_clean['Income'] > 0]
    
    # Ensure Age is reasonable
    df_clean = df_clean[(df_clean['Age'] >= 18) & (df_clean['Age'] <= 100)]
    
    logger.info(f"Data cleaning completed. Shape after cleaning: {df_clean.shape}")
    return df_clean


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical variables and scale numeric features.
    
    Args:
        df: Cleaned dataset
        
    Returns:
        Tuple[pd.DataFrame, dict]: Dataset with encoded features and label encoders
    """
    logger.info("Starting feature encoding...")
    
    df_encoded = df.copy()
    
    # Encode categorical variables using LabelEncoder
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_columns:
        if col != 'Credit_Score':  # Don't encode target yet
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
    
    # Scale numeric features (excluding target)
    numeric_columns = df_encoded.select_dtypes(include=[np.number]).columns
    if 'Credit_Score' in numeric_columns:
        numeric_columns = numeric_columns.drop('Credit_Score')
    
    scaler = StandardScaler()
    df_encoded[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns])
    
    # Encode target variable
    if 'Credit_Score' in df_encoded.columns:
        target_encoder = LabelEncoder()
        df_encoded['Credit_Score'] = target_encoder.fit_transform(df_encoded['Credit_Score'])
        label_encoders['Credit_Score'] = target_encoder
    
    logger.info("Feature encoding completed")
    return df_encoded, label_encoders


def split_data(df: pd.DataFrame, target_column: str = 'Credit_Score', 
               test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into training and testing sets.
    
    Args:
        df: Preprocessed dataset
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple containing X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting data with test_size={test_size}")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed_credit.csv") -> None:
    """
    Save processed dataset to file.
    
    Args:
        df: Processed dataset
        output_path: Path to save the processed data
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")


def preprocess_pipeline(data_path: str = "data/credit_score.csv", 
                       output_path: str = "data/processed_credit.csv") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Complete preprocessing pipeline.
    
    Args:
        data_path: Path to raw data
        output_path: Path to save processed data
        
    Returns:
        Tuple containing X_train, X_test, y_train, y_test, and label_encoders
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Load data
    df = load_data(data_path)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Encode features
    df_encoded, label_encoders = encode_features(df_clean)
    
    # Save processed data
    save_processed_data(df_encoded, output_path)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_encoded)
    
    logger.info("Preprocessing pipeline completed successfully")
    return X_train, X_test, y_train, y_test, label_encoders