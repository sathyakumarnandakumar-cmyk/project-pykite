from kiteconnect import KiteConnect
import os

# --- Retrieve your API Key and API Secret from environment variables ---
API_KEY = os.environ.get('KITE_API_KEY')
# The API_SECRET is generally not needed once you have the access_token,
# but it's good practice to have it available if you need to regenerate
# the access token.
# API_SECRET = os.environ.get('KITE_API_SECRET')

if not API_KEY:
    raise ValueError("KITE_API_KEY environment variable must be set.")

# --- How to get the access_token ---
# In a real application, you would store the access_token securely
# (e.g., in a database, a file, or an encrypted store) after the
# initial login and retrieve it here.
# For this example, let's assume you copy-pasted it from the Flask app output
# or retrieved it from a simple temporary file/environment variable after successful login.
ACCESS_TOKEN = os.environ.get('KITE_ACCESS_TOKEN') # Set this after Flask app gives it

if not ACCESS_TOKEN:
    print("WARNING: KITE_ACCESS_TOKEN not set. Please obtain it via the Flask app first.")
    print("Run `python app.py`, go to http://localhost:5000, log in, and copy the access token.")
    print("Then set it as an environment variable: export KITE_ACCESS_TOKEN=\"your_token_here\"")
    exit() # Exit if no access token is available

# Initialize KiteConnect with your API Key and the obtained Access Token
try:
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)

    print("KiteConnect instance initialized successfully with access token.")

    # --- Now you can make API calls ---

    # 1. Get user profile
    print("\n--- User Profile ---")
    try:
        profile = kite.profile()
        print(f"User Name: {profile['user_name']}")
        print(f"Email: {profile['email']}")
        print(f"Brokerage Plan: {profile['brokerage_plan']}")
    except Exception as e:
        print(f"Error fetching profile: {e}")

    # 2. Get instrument quotes (e.g., Nifty 50, Reliance)
    print("\n--- Instrument Quotes ---")
    try:
        # Example instruments (use correct exchange and instrument token/symbol)
        # You can get instrument tokens from kite.instruments() or kite.instruments("NSE")
        # For demo, using common symbols which might vary
        quotes = kite.quote(['NSE:NIFTY 50', 'NSE:RELIANCE', 'BSE:TCS'])
        for instrument, data in quotes.items():
            print(f"\nInstrument: {instrument}")
            print(f"  Last Traded Price (LTP): {data.get('last_price')}")
            print(f"  Change: {data.get('net_change')}")
            print(f"  Open: {data.get('ohlc', {}).get('open')}")
            print(f"  High: {data.get('ohlc', {}).get('high')}")
            print(f"  Low: {data.get('ohlc', {}).get('low')}")
            print(f"  Close: {data.get('ohlc', {}).get('close')}") # Previous day's close
    except Exception as e:
        print(f"Error fetching quotes: {e}")


    # 3. Get positions (holdings, intraday positions)
    print("\n--- Positions ---")
    try:
        positions = kite.positions()
        print("Net Positions:")
        for pos in positions.get('net', []):
            print(f"  {pos['tradingsymbol']}: Quantity={pos['quantity']}, P&L={pos['pnl']:.2f}")
        print("\nDay Positions:")
        for pos in positions.get('day', []):
            print(f"  {pos['tradingsymbol']}: Quantity={pos['quantity']}, P&L={pos['pnl']:.2f}")
    except Exception as e:
        print(f"Error fetching positions: {e}")

    # 4. Get order book
    print("\n--- Order Book ---")
    try:
        orders = kite.orders()
        if orders:
            print(f"Total {len(orders)} orders.")
            for order in orders[:min(3, len(orders))]: # Print first 3 orders
                print(f"  Order ID: {order['order_id']}, Symbol: {order['tradingsymbol']}, Type: {order['transaction_type']}, Status: {order['status']}, Price: {order['price']}")
        else:
            print("No orders found.")
    except Exception as e:
        print(f"Error fetching orders: {e}")

    # --- Other common operations (commented out, for you to explore) ---
    # Place an order (BE CAREFUL WITH LIVE TRADING!)
    # try:
    #     order_id = kite.place_order(
    #         tradingsymbol="RELIANCE",
    #         exchange=kite.EXCHANGE_NSE,
    #         transaction_type=kite.TRANSACTION_TYPE_BUY,
    #         quantity=1,
    #         order_type=kite.ORDER_TYPE_MARKET,
    #         product=kite.PRODUCT_CNC,
    #         variety=kite.VARIETY_REGULAR
    #     )
    #     print(f"Order placed successfully. Order ID: {order_id}")
    # except Exception as e:
    #     print(f"Error placing order: {e}")

    # Get historical data
    # from datetime import datetime, timedelta
    # historical_data = kite.historical_data(
    #     instrument_token=256265, # Example: Reliance token (get this from instruments API)
    #     from_date=datetime.now() - timedelta(days=7),
    #     to_date=datetime.now(),
    #     interval="day"
    # )
    # print("\n--- Historical Data (first 3 days) ---")
    # for data in historical_data[:3]:
    #     print(data)

except Exception as e:
    print(f"An error occurred during KiteConnect initialization or API call: {e}")