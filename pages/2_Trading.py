import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

st.title("⚡ Trading Terminal")

# --- 1. Instrument Search ---
search_query = st.text_input("Search Instrument (e.g., INFY, RELIANCE, NIFTY)", "")

if search_query:
    # Use the optimized search endpoint we created
    response = requests.get(f"{API_BASE_URL}/search/instruments", params={"query": search_query})
    
    if response.status_code == 200:
        results = response.json().get("results", [])
        
        if results:
            # Create a selection list
            options = {f"{i['exchange']}:{i['tradingsymbol']}": i for i in results}
            selected_symbol = st.selectbox("Select precise instrument", options.keys())
            
            if st.button("Open Order Window"):
                place_order_dialog(options[selected_symbol])
        else:
            st.warning("No instruments found.")

# --- 2. Order Dialog Box ---
@st.dialog("Place Order")
def place_order_dialog(instrument):
    st.subheader(f"{instrument['exchange']}:{instrument['tradingsymbol']}")
    st.write(f"Instrument Token: {instrument['instrument_token']}")
    
    col1, col2 = st.columns(2)
    with col1:
        transaction_type = st.radio("Action", ["BUY", "SELL"])
        quantity = st.number_input("Quantity", min_value=1, step=1)
    with col2:
        product = st.selectbox("Product", ["MIS", "CNC", "NRML"])
        order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"])

    price = 0
    if order_type == "LIMIT" or order_type == "SL":
        price = st.number_input("Price", min_value=0.05, step=0.05)

    if st.button("Submit Order", use_container_width=True):
        # Prepare payload for the FastAPI place_order endpoint
        payload = {
            "symbol": instrument['tradingsymbol'],
            "exchange": instrument['exchange'],
            "transaction_type": transaction_type,
            "quantity": quantity,
            "product": product,
            "order_type": order_type
        }
        
        try:
            # Calling the /orders/place POST endpoint
            resp = requests.post(f"{API_BASE_URL}/orders/place", params=payload)
            if resp.status_code == 200:
                st.success(f"Order Placed! ID: {resp.json().get('order_id')}")
                if st.button("Close"):
                    st.rerun()
            else:
                st.error(f"Error: {resp.json().get('detail')}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")
