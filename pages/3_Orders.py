
import streamlit as st
import requests
import pandas as pd

# The base URL for your FastAPI backend
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Order Management", layout="wide")
st.title("📋 Orders & Trades")

# Use tabs to separate the Order Book from the Trade Book
tab1, tab2 = st.tabs(["Order Book", "Trade Book"])

with tab1:
    st.subheader("Today's Orders")
    try:
        # Fetching all orders from the /orders endpoint
        order_resp = requests.get(f"{API_BASE_URL}/orders")
        
        if order_resp.status_code == 200:
            orders = order_resp.json()
            if orders:
                df_orders = pd.DataFrame(orders)
                
                # Displaying key columns for better readability
                cols = ["order_id", "tradingsymbol", "transaction_type", "status", "quantity", "price", "order_type"]
                st.dataframe(df_orders[cols], use_container_width=True)
                
                # --- Cancel Order Section ---
                st.divider()
                st.subheader("Cancel Pending Order")
                
                # Filter for orders that can actually be cancelled (OPEN or TRIGGER PENDING)
                pending_orders = [o['order_id'] for o in orders if o['status'] in ['OPEN', 'TRIGGER PENDING']]
                
                if pending_orders:
                    order_to_cancel = st.selectbox("Select Order ID to Cancel", pending_orders)
                    if st.button("Cancel Order", type="primary"):
                        # Calling the /orders/{order_id}/cancel POST endpoint
                        cancel_resp = requests.post(f"{API_BASE_URL}/orders/{order_to_cancel}/cancel")
                        if cancel_resp.status_code == 200:
                            st.success(f"Order {order_to_cancel} cancelled successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to cancel order.")
                else:
                    st.info("No cancellable pending orders found.")
            else:
                st.write("No orders placed today.")
    except Exception as e:
        st.error(f"Error fetching orders: {e}")

with tab2:
    st.subheader("Executed Trades")
    try:
        # Fetching all executed trades from the /trades endpoint
        trade_resp = requests.get(f"{API_BASE_URL}/trades")
        
        if trade_resp.status_code == 200:
            trades = trade_resp.json()
            if trades:
                df_trades = pd.DataFrame(trades)
                # Displaying key trade details
                trade_cols = ["trade_id", "order_id", "tradingsymbol", "transaction_type", "quantity", "average_price", "fill_timestamp"]
                st.dataframe(df_trades[trade_cols], use_container_width=True)
            else:
                st.write("No trades executed today.")
    except Exception as e:
        st.error(f"Error fetching trades: {e}")