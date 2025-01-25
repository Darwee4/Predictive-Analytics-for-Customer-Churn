# Customer Churn Prediction System

## Overview
This project implements a machine learning solution for predicting customer churn using the Telco Customer Churn dataset. The system includes data preprocessing, feature engineering, model training with XGBoost, and a Streamlit-based dashboard for visualization and predictions.

## Key Features
- Automated data preprocessing and feature engineering
- XGBoost model with ROC-AUC evaluation
- Interactive Streamlit dashboard
- Visualizations using Plotly
- Feature importance analysis
- Real-time churn probability predictions

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Darwee4/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```

2. Upload your Telco Customer Churn dataset (CSV format) through the sidebar

3. Explore the dashboard:
- View model performance metrics
- Analyze feature importance
- Make real-time predictions

## Model Details
- Algorithm: XGBoost Classifier
- Evaluation Metric: ROC-AUC Score
- Key Features:
  - Tenure bins
  - Monthly charges to income ratio
  - Standardized numerical features
  - Encoded categorical variables

## Requirements
- Python 3.8+
- See requirements.txt for full dependency list

## License
This project is licensed under the MIT License - see the LICENSE file for details.
