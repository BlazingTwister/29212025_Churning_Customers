import streamlit as st
import pickle
import pandas as pd
from keras.models import load_model

#Loading the scaler, label encoder, and Keras model
with open('scaler.pkl', 'rb') as s_file:
    scaler = pickle.load(s_file)

with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

keras_model = load_model('Keras_Model.h5')

#Categorical columns
categorical_columns = ['gender', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']

#Numerical columns
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Mapping dictionaries for label encoding
gender_mapping = {'Male': 0, 'Female': 1}
internet_service_mapping = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
online_security_mapping = {'No': 0, 'Yes': 1, 'No internet service': 2}
online_backup_mapping = {'No': 0, 'Yes': 1, 'No internet service': 2}
tech_support_mapping = {'No': 0, 'Yes': 1, 'No internet service': 2}
contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
paperless_billing_mapping = {'No': 0, 'Yes': 1}
payment_method_mapping = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}


#The Streamlit app
def main():
    st.title("Customer Churn Prediction App")

    #Collecting user input
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

    # Manual label encoding
    user_input['gender'] = gender_mapping.get(user_input['gender'], -1)
    user_input['InternetService'] = internet_service_mapping.get(user_input['InternetService'], -1)
    user_input['OnlineSecurity'] = online_security_mapping.get(user_input['OnlineSecurity'], -1)
    user_input['OnlineBackup'] = online_backup_mapping.get(user_input['OnlineBackup'], -1)
    user_input['TechSupport'] = tech_support_mapping.get(user_input['TechSupport'], -1)
    user_input['Contract'] = contract_mapping.get(user_input['Contract'], -1)
    user_input['PaperlessBilling'] = paperless_billing_mapping.get(user_input['PaperlessBilling'], -1)
    user_input['PaymentMethod'] = payment_method_mapping.get(user_input['PaymentMethod'], -1)


    # Scale numerical features using the saved scaler
    numerical_values = [user_input[column] for column in numerical_columns]
    scaled_values = scaler.transform([numerical_values])
    scaled_values = scaled_values.flatten()  # Flatten the result to get a 1D array
    
    # Update user_input with scaled numerical values
    for column, scaled_value in zip(numerical_columns, scaled_values):
        user_input[column] = scaled_value
        
    # Churn button
    if st.button("Churn"):
        prediction = keras_model.predict(user_input)
        
        # Assuming prediction is an array, get the confidence level
        confidence_level = prediction[0]

        # Display the confidence level
        st.subheader("Confidence Level:")
        st.write(f"The confidence level is {confidence_level:.2%}")

        # Displaying the prediction
        st.subheader("Prediction:")
        if prediction > 0.5:
            st.error("Churn: Customer is likely to churn.")
        else:
            st.success("No Churn: Customer is likely to stay.")


if __name__ == '__main__':
    main()
    
