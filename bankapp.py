import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/KANYIANALYST/BankMarketingapp/main/bank-full.csv'
    data = pd.read_csv(url)
    return data

data = load_data()

st.title('Bank Marketing Campaign Analysis')
st.write('## Data Overview')
st.write(data.head())

# Feature Importance Visualization
# Uncomment the below lines after defining best_rf in your code
# st.write('## Feature Importance')
# fig, ax = plt.subplots()
# sns.barplot(y=data.columns[:-1], x=best_rf.feature_importances_, ax=ax)
# st.pyplot(fig)
