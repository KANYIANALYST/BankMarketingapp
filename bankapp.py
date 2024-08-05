import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import urllib.request

# Load data
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/KANYIANALYST/BankMarketingapp/main/bank-full-encoded.csv'
    data = pd.read_csv(url)
    return data

data = load_data()

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    url = 'https://github.com/KANYIANALYST/BankMarketingapp/raw/main/best_rf_model.pkl'
    urllib.request.urlretrieve(url, 'best_rf_model.pkl')
    model = joblib.load('best_rf_model.pkl')
    return model

best_rf = load_model()

st.title('Bank Marketing Campaign Analysis')
st.write('## Data Overview')
st.write(data.head())

# Feature Importance Visualization
st.write('## Feature Importance')
fig, ax = plt.subplots()
sns.barplot(y=data.columns[:-1], x=best_rf.feature_importances_, ax=ax)
st.pyplot(fig)
