import streamlit as st
import pandas as pd

# Load the CSV file
df = pd.read_csv('feature_importance.csv')
import matplotlib.pyplot as plt
import seaborn as sns

# Sort the features by importance
df = df.sort_values(by='importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=df)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Display the plot in Streamlit
st.pyplot(plt)

import plotly.express as px

# Sort the features by importance
df = df.sort_values(by='importance', ascending=False)

# Create a bar chart
fig = px.bar(df, x='importance', y='feature', orientation='h', 
             title='Feature Importances')

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
# Add a slider to filter the minimum importance value
min_importance = st.slider('Minimum Importance', min_value=0.0, max_value=1.0, value=0.0)

# Filter the dataframe based on the slider value
filtered_df = df[df['importance'] >= min_importance]

# Plot the filtered feature importances
fig = px.bar(filtered_df, x='importance', y='feature', orientation='h', 
             title='Filtered Feature Importances')

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)
