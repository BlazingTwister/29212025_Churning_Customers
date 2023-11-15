Customer Churn Prediction App
Overview
This Project uses a Streamlit app utilizes a trained machine learning model to predict customer churn based on input variables. The model is built using a combination of categorical and numerical features, and it has been trained to make predictions on the likelihood of a customer churning.

Functionalities
User Input
The app allows users to input specific information related to a customer's profile, including:

Tenure: The length of time the customer has been with the service.
Gender: The gender of the customer.
Internet Service: The type of internet service the customer is using (DSL, Fiber optic, or No internet service).
Online Security, Online Backup, Tech Support: Availability of these services for the customer.
Contract: The duration of the customer's contract (Month-to-month, One year, Two years).
Paperless Billing: Whether the customer opts for paperless billing (Yes or No).
Payment Method: The method the customer uses for payment.
Monthly Charges: The amount the customer pays monthly.
Total Charges: The total charges incurred by the customer.
Prediction

Upon inputting these details, the app processes the information through a trained model and provides a prediction on whether the customer is likely to churn. Additionally, the app displays the confidence level associated with the prediction.

Model Components
The app uses a trained machine learning model that includes a scaler, label encoder, and a Keras neural network model. These components are loaded from saved files (scaler.pkl, label_encoder.joblib, Keras_Model.h5) during the app's initialization.
