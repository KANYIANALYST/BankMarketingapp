import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from urllib.request import urlopen

# Load data
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/KANYIANALYST/BankMarketingapp/main/bank-full-encoded.csv'
    data = pd.read_csv(url)
    return data

# Load feature importance
@st.cache
def load_feature_importance():
    url = 'https://raw.githubusercontent.com/KANYIANALYST/BankMarketingapp/main/feature_importance.csv'
    feature_importance_df = pd.read_csv(url)
    return feature_importance_df

# Load model
@st.cache
def load_model():
    url = 'https://github.com/KANYIANALYST/BankMarketingapp/blob/main/best_rf_model.pkl?raw=true'
    model = joblib.load(urlopen(url))
    return model

data = load_data()
feature_importance_df = load_feature_importance()
best_rf = load_model()

st.title('Bank Marketing Campaign Analysis')
st.write('## Data Overview')
st.write(data.head())

# Ensure the feature importances length matches the number of features
num_features = len(data.columns) - 1  # Exclude the target variable column
feature_importances_length = len(feature_importance_df)

if num_features != feature_importances_length:
    st.error(f"Mismatch in lengths: Number of features in dataset ({num_features}) does not match length of feature importances ({feature_importances_length}).")
else:
    # Feature Importance Visualization
    st.write('## Feature Importance')
    fig, ax = plt.subplots()
    sns.barplot(y=data.columns[:-1], x=feature_importance_df['importance'], ax=ax)
    st.pyplot(fig)
