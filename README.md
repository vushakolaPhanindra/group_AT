# 💳 Credit Score Intelligence

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120.0-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered credit score prediction and explanation system that combines machine learning with explainable AI to provide transparent, actionable insights for credit decisions.

## 📋 Overview

Credit Score Intelligence is a comprehensive solution that leverages Random Forest classification, SHAP (SHapley Additive exPlanations) values, and Large Language Models to predict credit scores while providing human-readable explanations. The system consists of a FastAPI backend for ML inference and a Streamlit frontend for interactive user experience.

### Key Features

- 🎯 **Accurate Predictions**: Random Forest model trained on comprehensive financial features
- 🧠 **Explainable AI**: SHAP values and LLM-generated rationales for transparent decision-making
- 📊 **Interactive Dashboards**: Real-time visualizations with Plotly and Matplotlib
- 🚀 **Production Ready**: FastAPI backend with automatic API documentation
- 💻 **User-Friendly**: Intuitive Streamlit interface with responsive design
- 🔧 **Modular Architecture**: Clean separation of concerns for easy maintenance

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Streamlit UI] --> B[Interactive Forms]
        A --> C[Visual Dashboards]
        A --> D[Real-time Results]
    end
    
    subgraph "API Layer"
        E[FastAPI Server] --> F[/predict endpoint]
        E --> G[/explain endpoint]
        E --> H[Health Checks]
    end
    
    subgraph "ML Pipeline"
        I[Data Preprocessing] --> J[Feature Engineering]
        J --> K[Random Forest Model]
        K --> L[SHAP Analysis]
        L --> M[LLM Rationale Generation]
    end
    
    subgraph "Data Layer"
        N[Credit Dataset] --> O[Processed Data]
        O --> P[Trained Model]
        P --> Q[SHAP Values]
    end
    
    A --> E
    E --> I
    I --> N
    M --> A
    Q --> A
    
    style A fill:#ff6b6b
    style E fill:#4ecdc4
    style I fill:#45b7d1
    style N fill:#96ceb4
```

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **scikit-learn** - Machine learning library for Random Forest
- **SHAP** - Explainable AI for model interpretability
- **LangChain** - LLM framework for rationale generation
- **OpenAI GPT** - Large Language Model for explanations
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing

### Frontend
- **Streamlit** - Rapid web app development
- **Plotly** - Interactive visualizations
- **Matplotlib** - Static plotting
- **requests** - HTTP client for API communication

### Infrastructure
- **uvicorn** - ASGI server for FastAPI
- **joblib** - Model serialization
- **seaborn** - Statistical data visualization

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/credit-score-intelligence.git
   cd credit-score-intelligence
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional)
   ```bash
   # For OpenAI integration
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. **Run the complete pipeline**
   ```bash
   # Train the model and generate explanations
   python main.py
   ```

5. **Start the API server**
   ```bash
   cd src
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Launch the web interface**
   ```bash
   cd ui
   streamlit run app.py
   ```

7. **Access the application**
   - Web UI: http://localhost:8501
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

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

## 🔄 How It Works

### Step 1: Data Preprocessing
- Load and clean the credit score dataset
- Handle missing values and outliers
- Encode categorical variables (Gender, Education, Marital Status)
- Split data into training and testing sets (80/20)

### Step 2: Model Training
- Train Random Forest Classifier with hyperparameter tuning
- Perform cross-validation for robust evaluation
- Generate feature importance rankings
- Save trained model for production use

### Step 3: SHAP Analysis
- Compute SHAP values for model interpretability
- Generate summary plots showing feature importance
- Create waterfall plots for individual predictions
- Export SHAP data for further analysis

### Step 4: LLM Rationale Generation
- Process SHAP values into human-readable format
- Use LangChain with OpenAI to generate explanations
- Provide actionable financial recommendations
- Save rationales with timestamps

### Step 5: API Serving
- FastAPI endpoints for prediction and explanation
- Real-time model inference
- Automatic API documentation
- Health monitoring and error handling

### Step 6: Web Interface
- Interactive Streamlit dashboard
- Real-time form validation
- Dynamic visualization updates
- Responsive design for all devices

## 📊 Example Output

### Generated Rationale

```
# Credit Score Analysis Report

## Prediction: Good

Based on the analysis, your credit score is predicted to be **Good**.

The overall assessment is positive, indicating strong creditworthiness.

**Key positive factor**: Income contributes significantly (+0.156) to your credit score.

**Area for improvement**: Number_of_Loans has a negative impact (-0.023) on your credit score.

**Recommendations:**
- Maintain consistent payment history
- Keep credit utilization below 30%
- Monitor your credit report regularly
- Consider diversifying your credit mix

## Technical Details
- Analysis based on SHAP (SHapley Additive exPlanations) values
- Model confidence: High
- Data points analyzed: 14

---
*This analysis is generated by our AI-powered credit scoring system.*
```

### API Response Example

```json
{
  "category": "Good",
  "confidence": 0.847,
  "feature_importance": {
    "Income": 0.156,
    "Credit_Utilization_Ratio": 0.134,
    "Credit_History_Length": 0.098,
    "Interest_Rate": 0.087,
    "Age": 0.076
  },
  "rationale": "Your credit score is predicted to be **Good**...",
  "shap_plot": "outputs/plots/sample_explain_20241025_143022.png"
}
```

## 🎯 Visual Dashboards

### Feature Correlation Heatmap
Interactive heatmap showing relationships between financial features, helping identify patterns and dependencies.

### SHAP Feature Importance
Dynamic bar chart displaying the most influential factors in credit score prediction with real-time updates.

### Confusion Matrix
Visual representation of model performance across different credit score categories with precision and recall metrics.

## 🔮 Future Improvements

### Short Term
- [ ] **Real-time Data Integration**: Connect to live financial data sources
- [ ] **Multi-model Support**: Add XGBoost, LightGBM, and Neural Networks
- [ ] **Advanced Visualizations**: 3D plots and interactive dashboards
- [ ] **User Authentication**: Secure user accounts and data privacy
- [ ] **Mobile App**: React Native mobile application

### Medium Term
- [ ] **Federated Learning**: Privacy-preserving model training
- [ ] **A/B Testing Framework**: Compare different model versions
- [ ] **Automated Retraining**: CI/CD pipeline for model updates
- [ ] **Multi-language Support**: Internationalization for global users
- [ ] **Advanced Analytics**: Time series analysis and trend prediction

### Long Term
- [ ] **Blockchain Integration**: Immutable credit score records
- [ ] **IoT Data Sources**: Real-time spending and income tracking
- [ ] **Quantum ML**: Next-generation quantum machine learning
- [ ] **Global Deployment**: Multi-region cloud infrastructure
- [ ] **Regulatory Compliance**: GDPR, CCPA, and financial regulations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Credit Score Intelligence

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```



<div align="center">
  <strong>Built with ❤️ for transparent and fair credit decisions</strong>
</div>
