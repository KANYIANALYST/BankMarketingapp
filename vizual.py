import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page configuration
st.set_page_config(page_title="Feature Importance Visualization", layout="wide")

# Title of the app
#st.title("Feature Importance Visualization")

# Load the feature importance data
url = "https://github.com/KANYIANALYST/BankMarketingapp/raw/main/feature_importance.csv"
data = pd.read_csv(url)

# Display the data
st.write("### Feature Importance Data", data)

# Rename columns if needed
# Adjust these names based on actual column names in your CSV file
data.rename(columns={
    'existing_feature_name': 'Feature',  # Replace 'existing_feature_name' with actual column name
    'existing_importance_name': 'Importance'  # Replace 'existing_importance_name' with actual column name
}, inplace=True)

# Check the DataFrame structure after renaming
# st.write("### Renamed Feature Importance Data", data)

# Plot feature importance
st.write("### Feature Importance Bar Chart")
if not data.empty:
    if 'Feature' in data.columns and 'Importance' in data.columns:
        # Plot the data
        plt.figure(figsize=(11, 10))
        sns.barplot(x='Importance', y='Feature', data=data, palette='viridis')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        st.pyplot(plt)
    else:
        st.error("The DataFrame does not contain 'Feature' and 'Importance' columns.")
else:
    st.error("The CSV file is empty or could not be loaded.")

# Add a footer or additional info if necessary
st.write("---")
st.write("Visualization created by [Esther Kanyi](https://github.com/KANYIANALYST)")


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
