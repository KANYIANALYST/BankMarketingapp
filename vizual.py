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
    # Ensure that the necessary columns exist
    if 'feature' in data.columns and 'importance' in data.columns:
        # Plot the data
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=data, palette='viridis')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        st.pyplot(plt)
    else:
        st.error("The CSV file does not contain 'feature' and 'importance' columns.")
else:
    st.error("The CSV file is empty or could not be loaded.")

# Add a footer or additional info if necessary
st.write("---")
st.write("Visualization created by [Your Name](https://github.com/KANYIANALYST)")
