import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page configuration
#st.set_page_config(page_title="Feature Importance Visualization", layout="wide")

# Title of the app
st.title("Feature Importance Visualization")

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
