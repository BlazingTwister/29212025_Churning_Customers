import streamlit as st
import pickle
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder


#Loading the scaler, label encoder, and Keras model
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

keras_model = load_model('Keras_Model.h5')

#Categorical columns
categorical_columns = ['gender', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'TechSupport', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']


#preprocessing input data
def preprocessing_input(input):
    label_encoder = LabelEncoder()
    
    # Convert categorical features to numerical using label encoder
    for column in categorical_columns:
        input[column] = label_encoder.fit_transform([input[column]])

    # Scale the input using the saved scaler
    input_sc = scaler.transform(input)

    return input_sc


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

   
    user_input = pd.DataFrame({
        'tenure': [tenure],
        'gender': [gender],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'TechSupport': [tech_support],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })


    p_input = preprocessing_input(user_input)
    
    prediction = keras_model.predict(p_input)
    
    
    #Displaying the prediction and confidence
    st.subheader("Prediction:")
    if prediction == 1:
        st.error("Churn: Customer is likely to churn.")
    else:
        st.success("No Churn: Customer is likely to stay.")

if __name__ == '__main__':
    main()
    