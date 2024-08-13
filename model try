import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from urllib.request import urlopen


# Load the trained model
model = joblib.load('bank_marketing_model.pkl')

# Create input fields
st.title("Bank Marketing Campaign Prediction")
age = st.number_input("Age", min_value=18, max_value=100, step=1)
job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
balance = st.number_input("Average Yearly Balance (in euros)", min_value=0)
contact = st.selectbox("Contact Communication Type", ["telephone", "cellular", "unknown"])
day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, step=1)
month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
duration = st.number_input("Last Contact Duration (in seconds)", min_value=0)

# Additional inputs based on your model's needs...

# Prediction button
if st.button("Predict Subscription Likelihood"):
    # Assuming the model needs a dataframe for prediction
    input_data = pd.DataFrame({
        "age": [age], "job": [job], "marital": [marital], "education": [education],
        "balance": [balance], "contact": [contact], "day": [day], "month": [month],
        "duration": [duration]
        # Include all necessary fields here
    })

    prediction = model.predict(input_data)
    st.write(f"The likelihood of subscription is: {'Yes' if prediction[0] == 1 else 'No'}")
