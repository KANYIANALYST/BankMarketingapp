import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from urllib.request import urlopen

# Load model
@st.cache_data 
def load_model():
    url = 'https://github.com/KANYIANALYST/BankMarketingapp/raw/main/best_rf_model.pkl'
    model = joblib.load(urlopen(url))
    return model

# Load the model
model = load_model()

# Create input fields
st.title("Bank Marketing Campaign Prediction")

# Input fields for the features used in the model
age = st.number_input("Age", min_value=18, max_value=100, step=1)
balance = st.number_input("Average Yearly Balance (in euros)", min_value=0)
day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, step=1)
duration = st.number_input("Last Contact Duration (in seconds)", min_value=0)
campaign = st.number_input("Number of Contacts Performed During Campaign", min_value=1)
pdays = st.number_input("Days Since Last Contact", min_value=-1)
previous = st.number_input("Number of Contacts Performed Before Campaign", min_value=0)
balance_duration_ratio = st.number_input("Balance/Duration Ratio", min_value=0.0)

# Categorical inputs
job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default_yes = st.selectbox("Has Credit in Default?", ["yes", "no"])
housing_yes = st.selectbox("Has Housing Loan?", ["yes", "no"])
loan_yes = st.selectbox("Has Personal Loan?", ["yes", "no"])
contact = st.selectbox("Contact Communication Type", ["telephone", "cellular", "unknown"])
month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
poutcome = st.selectbox("Outcome of Previous Campaign", ["unknown", "other", "failure", "success"])

# Map inputs to match the encoded columns
input_data = pd.DataFrame({
    "age": [age],
    "balance": [balance],
    "day": [day],
    "duration": [duration],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "balance_duration_ratio": [balance_duration_ratio],
    f"job_{job}": [1],
    f"marital_{marital}": [1],
    f"education_{education}": [1],
    "default_yes": [1 if default_yes == "yes" else 0],
    "housing_yes": [1 if housing_yes == "yes" else 0],
    "loan_yes": [1 if loan_yes == "yes" else 0],
    f"contact_{contact}": [1],
    f"month_{month}": [1],
    f"poutcome_{poutcome}": [1]
})

# Fill in the missing columns with zeros (because the model expects all possible encoded columns)
expected_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 
                    'balance_duration_ratio', 'job_blue-collar', 'job_entrepreneur', 
                    'job_housemaid', 'job_management', 'job_retired', 'job_self-employed', 
                    'job_services', 'job_student', 'job_technician', 'job_unemployed', 
                    'job_unknown', 'marital_married', 'marital_single', 'education_secondary', 
                    'education_tertiary', 'education_unknown', 'default_yes', 'housing_yes', 
                    'loan_yes', 'contact_telephone', 'contact_unknown', 'month_aug', 'month_dec', 
                    'month_feb', 'month_jan', 'month_jul', 'month_jun', 'month_mar', 'month_may', 
                    'month_nov', 'month_oct', 'month_sep', 'poutcome_other', 'poutcome_success', 
                    'poutcome_unknown']

# Add any missing columns with a default value of 0
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Ensure the input data has the same order as expected by the model
input_data = input_data[expected_columns]

# Debugging: Print the input data to check the format
st.write("Input Data:", input_data)

# Prediction
if st.button("Predict Subscription Likelihood"):
    try:
        prediction = model.predict(input_data)
        st.write(f"The likelihood of subscription is: {'Yes' if prediction[0] == 1 else 'No'}")
    except Exception as e:
        st.write(f"An error occurred: {e}")





# Feature importance bar chart
st.subheader("Feature Importance")
feature_importances = model.feature_importances_
features = expected_columns
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importances")
st.pyplot()

# Prediction summary
st.subheader("Prediction Summary")
prediction_counts = pd.Series(predictions).value_counts()
st.bar_chart(prediction_counts)

