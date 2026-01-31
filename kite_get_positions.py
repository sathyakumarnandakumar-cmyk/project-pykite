import pandas as pd
import requests
import io
from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv, find_dotenv

# --- Configuration ---
ENV_FILE_PATH = find_dotenv('./.env')

# Load environment variables
loaded = load_dotenv(ENV_FILE_PATH)
if not loaded:
    print("Warning: Environment variables could not be loaded from .env file (File might not exist yet).")

# Get Credentials
API_KEY = os.environ.get('KITE_API_KEY')
API_SECRET = os.environ.get('KITE_API_SECRET')
REQUEST_TOKEN = os.environ.get('KITE_REQUEST_TOKEN')

# 1. Initialize Kite Connect & Generate Session
kite = KiteConnect(api_key=API_KEY)

try:
    #kite_instance = kite.generate_session(REQUEST_TOKEN, api_secret=API_SECRET)
    kite.set_access_token(os.environ.get('KITE_ACCESS_TOKEN'))
    print("✅ Login successful!")
except Exception as e:
    print(f"❌ Error generating session: {e}")
    exit()

# --- Core Task ---

# 2. Get Master Instrument List (Directly to Pandas, No CSV)
print("Fetching master instrument list into memory...")
headers = {'X-Kite-Version': '3'}
url = "https://api.kite.trade/instruments"
response = requests.get(url, headers=headers)

if response.status_code == 200:
    # Read the CSV content from the response string directly into a DataFrame
    df_master_instruments = pd.read_csv(io.StringIO(response.text))
    
    # Optional: Filter for relevant segments to keep the DataFrame light (e.g., NSE/NFO)
    # df_master_instruments = df_master_instruments[df_master_instruments['exchange'].isin(['NSE', 'NFO'])]
    
    print(f"✅ Loaded {len(df_master_instruments)} instruments into local Pandas DataFrame.")
    print(df_master_instruments.head()) # Preview the data
else:
    print("❌ Failed to fetch instrument list")
    exit()

# 3. Get Current Positions
print("\nFetching current positions...")
positions_response = kite.positions()
net_positions = positions_response['net']

if not net_positions:
    print("No open positions found.")
else:
    # 4. Extract symbols for LTP fetching
    instruments_to_track = []
    
    # Create a mapping to easily look up instrument tokens or details if needed
    for pos in net_positions:
        symbol_key = f"{pos['exchange']}:{pos['tradingsymbol']}"
        instruments_to_track.append(symbol_key)

    # 5. Get LTP for all position instruments
    print(f"Fetching LTP for: {instruments_to_track}")
    ltp_data = kite.ltp(instruments_to_track)

    # 6. Merge LTP data into a Results DataFrame
    results = []
    for pos in net_positions:
        symbol_key = f"{pos['exchange']}:{pos['tradingsymbol']}"
        # safe_get ltp in case the symbol lookup fails
        last_price = ltp_data.get(symbol_key, {}).get('last_price', 0)
        
        # Calculate current value using the fetched LTP
        current_val = pos['quantity'] * last_price
        
        results.append({
            'Instrument': pos['tradingsymbol'],
            'Exchange': pos['exchange'],
            'Token': pos['instrument_token'],
            'Qty': pos['quantity'],
            'Buy Price': pos['buy_price'],
            'LTP': last_price,
            'Current Value': current_val
        })

    df_positions = pd.DataFrame(results)
    
    print("\n--- Positions Data (Ready for DB) ---")
    print(df_positions)