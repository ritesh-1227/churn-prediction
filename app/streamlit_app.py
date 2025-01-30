import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import joblib

# Add src to path for imports
current_dir = os.path.dirname(__file__)
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from data_prep import DataPreprocessor
from feature_eng import FeatureEngineer
from model import ChurnModel
from config import MODEL_PATH, DATA_PATH, MODEL_FILENAME

def load_models():
    print(f"Current working directory: {os.getcwd()}")
    model_file = os.path.join(MODEL_PATH, MODEL_FILENAME)
    print(f"Loading from: {model_file}")
    
    # Debug what's in the file
    artifacts = joblib.load(model_file)
    print(f"Type of loaded artifacts: {type(artifacts)}")
    print(f"Content structure: {artifacts if isinstance(artifacts, dict) else 'Not a dictionary'}")
    
    preprocessor = artifacts.get('preprocessor')
    model = artifacts.get('model')
    return preprocessor, model

def main():
    st.title("Customer Churn Prediction")
    
    # Load preprocessor and model
    preprocessor, model = load_models()
    
    # Create input form
    st.header("Customer Information")
    
    # Basic customer info
    tenure_months = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = monthly_charges * tenure_months
    
    # Services
    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
    streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    
    # Payment
    st.subheader("Payment Information")
    payment_method = st.selectbox("Payment Method", 
                                ['Electronic check', 'Mailed check', 
                                 'Bank transfer (automatic)', 'Credit card (automatic)'])
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    
    # Demographics
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ['Female', 'Male'])
    senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
    partner = st.selectbox("Partner", ['No', 'Yes'])
    dependents = st.selectbox("Dependents", ['No', 'Yes'])
    
    # Predict button
    if st.button("Predict Churn Probability"):
    # Create input dataframe with only the necessary features
        input_data = pd.DataFrame({
        'gender': [gender],
        'senior_citizen': [1 if senior_citizen == 'Yes' else 0],
        'partner': [partner],
        'dependents': [dependents],
        'tenure_months': [tenure_months],
        'phone_service': [phone_service],
        'multiple_lines': [multiple_lines],
        'internet_service': [internet_service],
        'online_security': [online_security],
        'online_backup': [online_backup],
        'device_protection': [device_protection],
        'tech_support': [tech_support],
        'streaming_tv': [streaming_tv],
        'streaming_movies': [streaming_movies],
        'contract': [contract],
        'paperless_billing': [paperless_billing],
        'payment_method': [payment_method],
        'monthly_charges': [monthly_charges],
        'total_charges': [total_charges]
    })
    
        try:
            # Preprocess the input data with is_training=False
            processed_data = preprocessor.preprocess(input_data, is_training=False)
            # Prepare features with is_training=False
            X, _ = preprocessor.prepare_features(processed_data, is_training=False)
            
            # Make prediction
            prediction_proba = model.predict_proba(X)
            churn_probability = prediction_proba[0][1]
            
            # Display results
            st.header("Prediction Results")
            st.write(f"Churn Probability: {churn_probability:.2%}")
            
            # Risk level
            if churn_probability < 0.3:
                risk_level = "Low Risk"
                color = "green"
            elif churn_probability < 0.6:
                risk_level = "Medium Risk"
                color = "orange"
            else:
                risk_level = "High Risk"
                color = "red"
                
            st.markdown(f"Risk Level: <span style='color:{color}'>{risk_level}</span>", 
                       unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Full error details:")
            st.error(e)

if __name__ == "__main__":
    main()