import streamlit as st

# Function to redirect to another app
def redirect():
    st.markdown(
        '<a href="https://www.example.com" target="_blank">Go to Another App</a>',
        unsafe_allow_html=True
    )

# Main app
st.title("Main Streamlit App")
st.write("Welcome to the main app!")

if st.button("Go to Another App"):
    redirect()
