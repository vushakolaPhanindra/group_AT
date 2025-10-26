# 💳 Credit Score Intelligence

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120.0-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI-powered system for predicting credit scores and generating explainable insights.

## Overview

Predicts credit scores (**Good**, **Standard**, **Poor**) using Random Forest, SHAP, and LLM-based explanations. Includes a **FastAPI backend** and a **Streamlit frontend**.

## Features

- Accurate predictions using Random Forest  
- Explainable AI via SHAP and LLM-generated rationales  
- Interactive dashboards  
- User-friendly Streamlit interface  
- Modular architecture  

## Tech Stack

**Backend:** FastAPI, scikit-learn, SHAP, LangChain, OpenAI GPT, pandas, numpy  
**Frontend:** Streamlit, Plotly, Matplotlib  
**Infrastructure:** uvicorn, joblib, seaborn  

## Quick Start

### Prerequisites
- Python 3.11+  
- pip  

### Installation

```bash
git clone https://github.com/vushakolaPhanindra/credit-score-intelligence.git
cd credit-score-intelligence
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key-here"
python main.py
```
Run API

cd src
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Launch Web Interface
cd ui
```bash
streamlit run app.py
```

Web UI: http://localhost:8501

API Docs: http://localhost:8000/docs

## 📁 Project Structure

```
credit-score-intelligence/
├── 📁 src/                          # Backend source code
│   ├── 📄 api.py                    # FastAPI application
│   ├── 📄 preprocess.py             # Data preprocessing pipeline
│   ├── 📄 train_model.py            # Model training and evaluation
│   ├── 📄 explain_model.py          # SHAP analysis and visualization
│   ├── 📄 generate_rationale.py     # LLM-based explanation generation
│   └── 📄 utils.py                  # Utility functions
├── 📁 ui/                           # Frontend source code
│   └── 📄 app.py                    # Streamlit web application
├── 📁 data/                         # Data storage
│   ├── 📄 credit_score.csv          # Raw dataset
│   └── 📄 processed_credit.csv      # Cleaned dataset
├── 📁 models/                       # Model storage
│   └── 📄 credit_model.pkl          # Trained Random Forest model
├── 📁 outputs/                      # Generated outputs
│   ├── 📁 plots/                    # Visualization outputs
│   ├── 📁 shap_summaries/           # SHAP analysis data
│   └── 📁 rationales/               # Generated explanations
├── 📁 notebooks/                    # Jupyter notebooks
│   └── 📄 exploration.ipynb         # Data exploration
├── 📄 main.py                       # Main pipeline orchestrator
├── 📄 test_api.py                   # API testing script
├── 📄 requirements.txt              # Python dependencies
└── 📄 README.md                     # Project documentation
```
### API Response Sample:
```
{
  "category": "Good",
  "confidence": 0.847,
  "feature_importance": {"Income": 0.156, "Credit_Utilization_Ratio": 0.134},
  "rationale": "Your credit score is predicted to be Good"
}
```
### Contributing

Fork the repository

Create a branch (git checkout -b feature-name)

Commit changes (git commit -m "Add feature")

Push branch (git push origin feature-name)

Open a Pull Request
