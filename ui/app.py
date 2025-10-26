"""
Complete Interactive Streamlit Web App for Credit Score Intelligence

This module provides a comprehensive web interface for credit score prediction
and explanation using the FastAPI backend with SHAP values and LLM-generated rationales.
"""

import streamlit as st
import requests
import json
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Credit Score Intelligence",
    page_icon=":credit_card:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .error-card {
        background: linear-gradient(135deg, #f8d7da 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #dc3545;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Utility functions
@st.cache_data(ttl=300)
def check_api_health() -> bool:
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_predict_api(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call the predict API endpoint with error handling."""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=form_data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection Error: Cannot connect to API server."}
    except Exception as e:
        return {"error": f"Unexpected Error: {str(e)}"}

def call_explain_api(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """Call the explain API endpoint with error handling."""
    try:
        response = requests.post(f"{API_BASE_URL}/explain", json=form_data, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection Error: Cannot connect to API server."}
    except Exception as e:
        return {"error": f"Unexpected Error: {str(e)}"}

def display_prediction_result(result: Dict[str, Any]):
    """Display prediction results."""
    if "error" in result:
        st.markdown('<div class="error-card">', unsafe_allow_html=True)
        st.error(f"Error: {result['error']}")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    category = result.get('category', 'Unknown')
    confidence = result.get('confidence', 0)
    feature_importance = result.get('feature_importance', {})
    
    # Color coding for categories
    category_colors = {
        'Good': '🟢',
        'Standard': '🟡', 
        'Poor': '🔴'
    }
    
    icon = category_colors.get(category, '❓')
    
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.markdown(f"### {icon} Predicted Credit Score: **{category}**")
    st.markdown(f"**Confidence:** {confidence:.1%}")
    
    # Confidence bar
    st.progress(confidence)
    
    if feature_importance:
        st.markdown("**Top Contributing Factors:**")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5], 1):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"{i}. {feature.replace('_', ' ').title()}")
            with col2:
                st.progress(importance)
            with col3:
                st.markdown(f"{importance:.3f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_explanation_result(result: Dict[str, Any]):
    """Display explanation results."""
    if "error" in result:
        st.markdown('<div class="error-card">', unsafe_allow_html=True)
        st.error(f"Error: {result['error']}")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    category = result.get('category', 'Unknown')
    rationale = result.get('rationale', '')
    shap_plot = result.get('shap_plot', '')
    shap_data = result.get('shap_data', {})
    
    st.markdown(f"### AI Explanation for **{category}** Credit Score")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Rationale", "SHAP Analysis", "Feature Impact"])
    
    with tab1:
        st.markdown("#### Generated Rationale")
        if rationale:
            st.markdown(rationale)
        else:
            st.info("No rationale available")
    
    with tab2:
        st.markdown("#### SHAP Visualization")
        if shap_plot and os.path.exists(shap_plot):
            try:
                st.image(shap_plot, caption="SHAP Waterfall Plot", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying SHAP plot: {str(e)}")
        else:
            st.info("SHAP plot not available")
        
        # Try to show summary plot
        summary_plot_path = "outputs/plots/shap_summary.png"
        if os.path.exists(summary_plot_path):
            st.image(summary_plot_path, caption="SHAP Summary Plot", use_column_width=True)
    
    with tab3:
        st.markdown("#### Feature Impact Analysis")
        if shap_data:
            feature_values = shap_data.get('feature_values', {})
            shap_values = shap_data.get('shap_values', [])
            
            if feature_values and shap_values:
                impact_data = []
                for i, (feature, value) in enumerate(feature_values.items()):
                    if i < len(shap_values):
                        impact_value = shap_values[i] if isinstance(shap_values, list) else shap_values[0][i] if len(shap_values.shape) > 1 else shap_values[i]
                        impact_data.append({
                            'Feature': feature.replace('_', ' ').title(),
                            'Value': value,
                            'Impact': impact_value,
                            'Abs_Impact': abs(impact_value)
                        })
                
                if impact_data:
                    df_impact = pd.DataFrame(impact_data)
                    df_impact = df_impact.sort_values('Abs_Impact', ascending=False)
                    
                    st.markdown("**Top 10 Most Influential Features:**")
                    st.dataframe(df_impact.head(10)[['Feature', 'Value', 'Impact']], use_container_width=True)
                    
                    # Create interactive bar chart
                    plot_df = df_impact.head(10).copy()
                    plot_df['Color'] = plot_df['Impact'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
                    
                    fig = px.bar(
                        plot_df, 
                        x='Impact', 
                        y='Feature',
                        color='Color',
                        color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C'},
                        title="Feature Impact on Credit Score Prediction",
                        orientation='h'
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SHAP data available for visualization")

def create_user_input_form():
    """Create the user input form in the sidebar."""
    st.markdown("### Input Parameters")
    
    # Personal Information
    st.markdown("#### Personal Information")
    age = st.slider("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    dependents = st.slider("Number of Dependents", min_value=0, max_value=10, value=1)
    
    # Financial Information
    st.markdown("#### Financial Information")
    income = st.number_input("Annual Income ($)", min_value=0.0, value=75000.0, step=1000.0)
    credit_history = st.slider("Credit History Length (years)", min_value=0, max_value=50, value=8)
    interest_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=50.0, value=12.5, step=0.1)
    outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=15000.0, step=1000.0)
    
    # Credit Information
    st.markdown("#### Credit Information")
    credit_cards = st.slider("Number of Credit Cards", min_value=0, max_value=20, value=3)
    bank_accounts = st.slider("Number of Bank Accounts", min_value=0, max_value=10, value=2)
    loans = st.slider("Number of Active Loans", min_value=0, max_value=10, value=1)
    delay_days = st.slider("Days Delayed from Due Date", min_value=0, max_value=365, value=5)
    credit_utilization = st.slider("Credit Utilization Ratio", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    
    return {
        "Age": age,
        "Gender": gender,
        "Income": income,
        "Education": education,
        "Marital_Status": marital_status,
        "Number_of_Dependents": dependents,
        "Credit_History_Length": credit_history,
        "Number_of_Credit_Cards": credit_cards,
        "Number_of_Bank_Accounts": bank_accounts,
        "Interest_Rate": interest_rate,
        "Number_of_Loans": loans,
        "Delay_from_due_date": delay_days,
        "Outstanding_Debt": outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization
    }

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">Credit Score Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">AI-Powered Credit Score Prediction & Explanation</div>', unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("API server is not running. Please start the FastAPI server first.")
        st.info("To start the server, run: `cd src && uvicorn api:app --host 0.0.0.0 --port 8000 --reload`")
        st.warning("Running in Demo Mode - Some features may be limited")
        st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        form_data = create_user_input_form()
        
        st.markdown("### About This App")
        st.markdown("""
        **Credit Score Intelligence** is an AI-powered platform that combines machine learning with explainable AI to provide transparent credit score predictions.
        
        **Key Features:**
        - Accurate Predictions using Random Forest
        - Explainable AI with SHAP values
        - Interactive Visualizations
        - User-Friendly Interface
        
        **Technology Stack:**
        - Backend: FastAPI + scikit-learn
        - Frontend: Streamlit + Plotly
        - AI: SHAP + LangChain + OpenAI
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Quick Prediction")
        st.markdown("Get an instant credit score prediction based on your financial profile.")
        
        if st.button("Predict Credit Score", type="primary", use_container_width=True):
            with st.spinner("Analyzing your credit profile..."):
                result = call_predict_api(form_data)
                display_prediction_result(result)
    
    with col2:
        st.markdown("### Detailed Explanation")
        st.markdown("Get AI-powered explanations with SHAP analysis and feature importance.")
        
        if st.button("Get AI Explanation", type="secondary", use_container_width=True):
            with st.spinner("Generating AI explanation and SHAP analysis..."):
                result = call_explain_api(form_data)
                display_explanation_result(result)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Prediction Engine**
        - Random Forest Classifier
        - 14 Financial Features
        - Real-time Analysis
        """)
    
    with col2:
        st.markdown("""
        **Explanation Engine**
        - SHAP Values Analysis
        - LLM-Generated Rationale
        - Feature Importance
        """)
    
    with col3:
        st.markdown("""
        **Technology Stack**
        - FastAPI Backend
        - Streamlit Frontend
        - LangChain + OpenAI
        """)
    
    # API Status
    if check_api_health():
        st.success("API Server: Online")
    else:
        st.error("API Server: Offline")

if __name__ == "__main__":
    main()