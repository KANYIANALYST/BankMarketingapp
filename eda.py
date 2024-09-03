import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page configuration
st.set_page_config(page_title="EDA Visualization", layout="wide")

# Title of the app
st.title("Exploratory Data Analysis (EDA) Visualization")

# Load the dataset
url = "https://github.com/KANYIANALYST/BankMarketingapp/raw/main/bank-full-encoded.csv"
bank_full_data = pd.read_csv(url)

# Display dataset info and a preview
st.write("### Dataset Overview")
st.write("The dataset contains the following features:")
st.write(bank_full_data.info())
st.write("### Sample Data", bank_full_data.head())

# Distribution of numerical features
st.write("### Distribution of Numerical Features")
numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for i, feature in enumerate(numerical_features):
    ax = axes[i // 3, i % 3]
    bank_full_data[feature].hist(ax=ax, bins=20)
    ax.set_title(feature)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
fig.suptitle('Distribution of Numerical Features')
st.pyplot(fig)

# Distribution of categorical features
st.write("### Distribution of Categorical Features")
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
for feature in categorical_features:
    st.write(f"#### Distribution of {feature}")
    plt.figure(figsize=(10, 4))
    sns.countplot(data=bank_full_data, x=feature, order=bank_full_data[feature].value_counts().index)
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.clf()  # Clear the figure to avoid overlapping plots

# Footer or additional info if necessary
st.write("---")
st.write("Visualization created by [Esther Kanyi](https://github.com/KANYIANALYST)")
