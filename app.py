# Navigate to your repository
cd BankMarketingapp

# Create the app.py file (example content)
echo "
import streamlit as st
import pandas as pd

# Load data
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/KANYIANALYST/BankMarketingapp/main/Copy_of_Bank_Marketing_Prediction.ipynb'
    data = pd.read_csv(url)
    return data

data = load_data()

st.title('Bank Marketing Campaign Analysis')
st.write(data.head())
" > app.py

# Add, commit, and push the file
git add app.py
git commit -m "Add Streamlit app script"
git push
