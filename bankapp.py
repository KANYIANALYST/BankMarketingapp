import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from urllib.request import urlopen

# Load data
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/KANYIANALYST/BankMarketingapp/main/bank-full-encoded.csv'
    data = pd.read_csv(url)
    return data

# Load feature importance
@st.cache_data
def load_feature_importance():
    url = 'https://raw.githubusercontent.com/KANYIANALYST/BankMarketingapp/main/feature_importance.csv'
    feature_importance_df = pd.read_csv(url)
    return feature_importance_df

# Load model
@st.cache_data
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

# Exclude the target variable from the dataset columns
target_variable = 'Predicted Probability'  # Replace with the actual name of your target variable

data_features = [col for col in data.columns if col != target_variable]

# Debugging output
st.write('### Columns in the dataset:')
st.write(data_features)

st.write('### Feature importances length:')
st.write(len(feature_importance_df))

st.write('### Feature importances:')
st.write(feature_importance_df)

# Check for mismatch
if len(data_features) != len(feature_importance_df):
    st.error(f"Mismatch in lengths: Number of features in dataset ({len(data_features)}) does not match length of feature importances ({len(feature_importance_df)}).")
else:
    # Feature Importance Visualization
    st.write('## Feature Importance')
    fig, ax = plt.subplots()
    sns.barplot(y=data_features, x=feature_importance_df['importance'], ax=ax)
    st.pyplot(fig)
