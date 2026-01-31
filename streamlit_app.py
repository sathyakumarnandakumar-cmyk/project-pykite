import streamlit as st

st.set_page_config(page_title="Kite Connect Terminal", layout="wide")

st.title("🚀 Kite Connect Pro Terminal")
st.write("Welcome to your private trading dashboard. Use the sidebar to navigate.")

# Shared utility for API calls
API_BASE_URL = "http://127.0.0.1:8000" 

st.info("Ensure your FastAPI backend is running at " + API_BASE_URL)