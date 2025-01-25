import streamlit as st
import pandas as pd
import plotly.express as px
from churn_prediction import ChurnPredictor

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Title and description
st.title("Customer Churn Prediction Dashboard")
st.write("""
This dashboard provides insights into customer churn prediction using machine learning.
Explore model performance, feature importance, and make predictions.
""")

# Sidebar for file upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your Telco Customer Churn dataset", type=["csv"])

if uploaded_file is not None:
    # Initialize predictor
    predictor = ChurnPredictor(uploaded_file)
    
    # Train model and get metrics
    auc_score, conf_matrix = predictor.train_model()
    
    # Display metrics
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ROC-AUC Score", f"{auc_score:.3f}")
    with col2:
        st.write("Confusion Matrix:")
        st.plotly_chart(predictor.visualize_confusion_matrix(conf_matrix))
    
    # Feature importance
    st.header("Feature Importance")
    st.plotly_chart(predictor.visualize_feature_importance())
    
    # Prediction interface
    st.header("Make Predictions")
    st.write("Enter customer details to predict churn probability")
    
    # Create input fields
    col1, col2 = st.columns(2)
    with col1:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
        total_charges = st.number_input("Total Charges", min_value=0.0)
    with col2:
        tenure_bin = st.selectbox("Tenure Bin", options=[1, 2, 3, 4, 5])
        charges_to_income = st.number_input("Charges to Income Ratio", min_value=0.0)
    
    if st.button("Predict Churn Probability"):
        # Create input data frame
        input_data = pd.DataFrame({
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'tenure_bin': [tenure_bin],
            'charges_to_income': [charges_to_income]
        })
        
        # Make prediction
        prediction = predictor.model.predict_proba(input_data)[0][1]
        st.success(f"Churn Probability: {prediction:.2%}")
else:
    st.info("Please upload a dataset to begin analysis")
