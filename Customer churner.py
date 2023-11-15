#Customer churning 2.0

import streamlit as st
import pickle
import joblib
import pandas as pd
from keras.models import load_model
import numpy as np

# Load the scaler, label encoder, and Keras model
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

label_encoder = joblib.load('label_encoder.joblib')

keras_model = load_model('Keras_Model.h5')

# Define the categorical columns
categorical_columns = ['gender', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']

# Define the numerical columns
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

# a function to preprocess input data
def preprocessing_input(input):
    # Convert categorical features to numerical using label encoder
    for column in categorical_columns:
        input[column] = label_encoder.transform([input[column]])

    # Scale numerical features using the saved scaler
    input[numerical_columns] = scaler.transform([input[numerical_columns]])

    return input

#a function to make predictions
def make_prediction(input):
    # Preprocess the input
    input_data = preprocessing_input(input)

    # Make a prediction using the Keras model
    prediction_proba = keras_model.predict(np.array([input_data.values]))
    prediction = (prediction_proba[0, 0] > 0.5).astype(int)

    return prediction, prediction_proba[0, 0]

#the Streamlit app
def main():
    st.title("Customer Churn Prediction App")

    # Collect user input
    tenure = st.slider("Tenure", 0, 72, 1)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", ['No', 'Yes'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=20.0, step=1.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=0.0, step=1.0)

   
    user_input = {
        'tenure': tenure,
        'gender': gender,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'TechSupport': tech_support,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Make prediction
    prediction, confidence = make_prediction(pd.DataFrame([user_input]))

    # Display the prediction and confidence
    st.subheader("Prediction:")
    if prediction == 1:
        st.error("Churn: Customer is likely to churn.")
    else:
        st.success("No Churn: Customer is likely to stay.")

    st.subheader("Confidence Level:")
    st.write(f"{confidence * 100:.2f}%")
    
    