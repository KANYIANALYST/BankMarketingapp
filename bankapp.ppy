import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/KANYIANALYST/BankMarketingapp/main/bank-full-encoded.csv'
    data = pd.read_csv(url)
    return data

# Load model
@st.cache
def load_model():
    url = 'https://raw.githubusercontent.com/KANYIANALYST/BankMarketingapp/main/best_rf_model.pkl'
    model = joblib.load(url)
    return model

data = load_data()
best_rf = load_model()

st.title('Bank Marketing Campaign Analysis')
st.write('## Data Overview')
st.write(data.head())

# Feature Importance Visualization
st.write('## Feature Importance')
fig, ax = plt.subplots()

# Exclude the target variable and "Predicted Probability"
#data_features = data.drop(columns=['y', 'Predicted Probability']).columns

sns.barplot(y=data_features, x=best_rf.feature_importances_, ax=ax)
ax.set_title("Feature Importance")
ax.set_xlabel("Importance")
ax.set_ylabel("Features")
st.pyplot(fig)
