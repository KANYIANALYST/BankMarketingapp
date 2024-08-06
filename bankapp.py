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

# Get feature names from the dataset and feature importance data
data_features = list(data.columns[:-1])
importance_features = list(feature_importance_df['feature'])

# Check for mismatch
if len(data_features) != len(importance_features):
    st.error(f"Mismatch in lengths: Number of features in dataset ({len(data_features)}) does not match length of feature importances ({len(importance_features)}).")
    
    # Find and display the extra feature
    extra_features_in_data = set(data_features) - set(importance_features)
    extra_features_in_importance = set(importance_features) - set(data_features)
    
    if extra_features_in_data:
        st.write("Extra features in dataset:", extra_features_in_data)
    if extra_features_in_importance:
        st.write("Extra features in feature importance:", extra_features_in_importance)
else:
    # Feature Importance Visualization
    st.write('## Feature Importance')
    fig, ax = plt.subplots()
    sns.barplot(y=data_features, x=feature_importance_df['importance'], ax=ax)
    st.pyplot(fig)
