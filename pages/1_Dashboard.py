import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://127.0.0.1:8000"

st.title("📊 Portfolio Dashboard")

# Fetch Margins
margin_resp = requests.get(f"{API_BASE_URL}/user/margins")
if margin_resp.status_code == 200:
    equity = margin_resp.json().get('equity', {})
    st.metric("Available Cash", f"₹{equity.get('available', {}).get('cash', 0)}")

# Fetch Holdings
holdings_resp = requests.get(f"{API_BASE_URL}/portfolio/holdings")
if holdings_resp.status_code == 200:
    st.subheader("Your Holdings")
    df_holdings = pd.DataFrame(holdings_resp.json())
    st.dataframe(df_holdings, use_container_width=True)