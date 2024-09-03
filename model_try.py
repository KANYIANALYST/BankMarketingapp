import streamlit as st
import pandas as pd
import numpy as np
import joblib
from urllib.request import urlopen
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
@st.cache_data 
def load_model():
    url = 'https://github.com/KANYIANALYST/BankMarketingapp/raw/main/best_rf_model.pkl'
    model = joblib.load(urlopen(url))
    return model

# Load the model
model = load_model()

# Initialize LabelEncoders (same as before)
from sklearn.preprocessing import LabelEncoder

le_job = LabelEncoder()
le_marital = LabelEncoder()
le_education = LabelEncoder()

job_labels = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", 
              "self-employed", "services", "student", "technician", "unemployed", "unknown"]
le_job.fit(job_labels)

marital_labels = ["married", "single", "divorced"]
le_marital.fit(marital_labels)

education_labels = ["primary", "secondary", "tertiary", "unknown"]
le_education.fit(education_labels)

# Input fields
st.title("Bank Marketing Campaign Prediction")

# Numeric Inputs
age = st.number_input("Age", min_value=18, max_value=100, step=1)
balance = st.number_input("Average Yearly Balance (in euros)", min_value=0)
day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, step=1)
duration = st.number_input("Last Contact Duration (in seconds)", min_value=0)
campaign = st.number_input("Number of Contacts Performed During Campaign", min_value=1)
pdays = st.number_input("Days Since Last Contact", min_value=-1)
previous = st.number_input("Number of Contacts Performed Before Campaign", min_value=0)
balance_duration_ratio = st.number_input("Balance/Duration Ratio", min_value=0.0)

# Categorical Inputs
job = st.selectbox("Job", job_labels)
marital = st.selectbox("Marital Status", marital_labels)
education = st.selectbox("Education", education_labels)
default_yes = st.selectbox("Has Credit in Default?", ["yes", "no"])
housing_yes = st.selectbox("Has Housing Loan?", ["yes", "no"])
loan_yes = st.selectbox("Has Personal Loan?", ["yes", "no"])
contact = st.selectbox("Contact Communication Type", ["telephone", "cellular", "unknown"])
month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
poutcome = st.selectbox("Outcome of Previous Campaign", ["unknown", "other", "failure", "success"])

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    "age": [age],
    "balance": [balance],
    "day": [day],
    "duration": [duration],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "balance_duration_ratio": [balance_duration_ratio],
    "job": [le_job.transform([job])[0]],
    "marital": [le_marital.transform([marital])[0]],
    "education": [le_education.transform([education])[0]],
    "default_yes": [1 if default_yes == "yes" else 0],
    "housing_yes": [1 if housing_yes == "yes" else 0],
    "loan_yes": [1 if loan_yes == "yes" else 0],
    f"contact_{contact}": [1],
    f"month_{month}": [1],
    f"poutcome_{poutcome}": [1],
    f"job_{job}": [1],  
    f"marital_{marital}": [1],  
    f"education_{education}": [1],  
})

# Fill missing columns
expected_columns = ['job', 'marital', 'education', 'age', 'balance', 'day', 'duration',
                    'campaign', 'pdays', 'previous', 'balance_duration_ratio',
                    'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
                    'job_management', 'job_retired', 'job_self-employed', 'job_services',
                    'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
                    'marital_married', 'marital_single', 'education_secondary',
                    'education_tertiary', 'education_unknown', 'default_yes', 'housing_yes',
                    'loan_yes', 'contact_telephone', 'contact_unknown', 'month_aug',
                    'month_dec', 'month_feb', 'month_jan', 'month_jul', 'month_jun',
                    'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep',
                    'poutcome_other', 'poutcome_success', 'poutcome_unknown']

for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[expected_columns]

# Debug: Print the input data
st.write("Input Data:", input_data)

# Visualization: Histogram of Numeric Inputs
st.subheader("Input Data Visualization")
fig, axs = plt.subplots(2, 4, figsize=(15, 10))
sns.histplot(input_data['age'], ax=axs[0, 0], kde=True)
sns.histplot(input_data['balance'], ax=axs[0, 1], kde=True)
sns.histplot(input_data['day'], ax=axs[0, 2], kde=True)
sns.histplot(input_data['duration'], ax=axs[0, 3], kde=True)
sns.histplot(input_data['campaign'], ax=axs[1, 0], kde=True)
sns.histplot(input_data['pdays'], ax=axs[1, 1], kde=True)
sns.histplot(input_data['previous'], ax=axs[1, 2], kde=True)
sns.histplot(input_data['balance_duration_ratio'], ax=axs[1, 3], kde=True)

for ax in axs.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.tight_layout()
st.pyplot(fig)

# Prediction
if st.button("Predict Subscription Likelihood"):
    try:
        prediction = model.predict(input_data)
        st.subheader(f"The likelihood of subscription is: {'Yes' if prediction[0] == 1 else 'No'}")

        # Visualization: Gauge or Bar for Likelihood
        likelihood = model.predict_proba(input_data)[0][1] * 100
        st.subheader("Subscription Likelihood")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(["Subscription"], [likelihood], color="skyblue")
        ax.set_xlim(0, 100)
        ax.set_xlabel('Likelihood (%)')
        ax.set_title(f"{likelihood:.2f}%")
        st.pyplot(fig)

    except Exception as e:
        st.write(f"An error occurred: {e}")
