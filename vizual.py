import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page configuration
st.set_page_config(page_title="Feature Importance Visualization", layout="wide")

# Title of the app
st.title("Feature Importance Visualization")

# Load the feature importance data
url = "feature_importance.csv"
data = pd.read_csv(url)

# Display the data
st.write("### Feature Importance Data", data)

# Plot feature importance
st.write("### Feature Importance Bar Chart")
if not data.empty:
    # Check the actual column names in your CSV file
    if 'Name' in data.columns and 'Importance' in data.columns:
        # Plot the data
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Name', data=data, palette='viridis')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        st.pyplot(plt)
    else:
        st.error("The CSV file does not contain 'Name' and 'Importance' columns.")
else:
    st.error("The CSV file is empty or could not be loaded.")

# Add a footer or additional info if necessary
st.write("---")
st.write("Visualization created by [Esther Kanyi](https://github.com/KANYIANALYST)")
