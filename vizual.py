import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen


@st.cache_data
def load_model():
    url = 'https://github.com/KANYIANALYST/BankMarketingapp/raw/main/best_rf_model.pkl'
    model = joblib.load(urlopen(url))
    return model

model = load_model()

# Extract feature importances
importances = model.feature_importances_
features = ['job', 'marital', 'education', 'age', 'balance', 'day', 'duration',
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

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

st.title("Feature Importance Visualization")

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
st.pyplot(plt)
