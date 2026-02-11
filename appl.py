# ==============================================================================
# 1. IMPORTS & SETUP
# ==============================================================================
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date
import os
import tempfile
import warnings
import dash
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Volatility Imports
from py_vollib.black_scholes.implied_volatility import implied_volatility as bsm_iv

# --- IMPORT KITE ---
# We assume load_kite_from_access.py is in the same folder
try:
    import load_kite_from_access
    kite = load_kite_from_access.kite
    print(f"[OK] Connected to Kite: {kite.profile()['user_name']}")
except Exception as e:
    print(f"[X] Error importing Kite: {e}")
    print("Ensure 'load_kite_from_access.py' is in this folder and login is valid.")
    sys.exit(1) # Stop app if login fails

# ==============================================================================
# 2. GLOBAL DATA LOADING (Runs once when server starts)
# ==============================================================================
print("\n[..] Initializing: Fetching Instruments & Building Lookups... (This takes a moment)")

try:
    # 1. Fetch Instruments
    instruments_nse = kite.instruments("NSE")
    instruments_nfo = kite.instruments("NFO")
    instruments_mcx = kite.instruments("MCX")
    instruments_bse = kite.instruments("BSE")
    instruments_bfo = kite.instruments("BFO")
    
    instruments_all = instruments_nse + instruments_nfo + instruments_mcx + instruments_bse + instruments_bfo
    df_all = pd.DataFrame(instruments_all)
    
    # 2. Standardize Data (Helper function included here for speed)
    def standardize_dataframe(df):
        df_clean = df.copy()
        for col in ['expiry', 'last_date']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        for col in ['strike', 'tick_size', 'lot_size']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        return df_clean

    df_all = standardize_dataframe(df_all)
    
    # Separate MCX and BFO for specific logic
    df_mcx = df_all[df_all['exchange'] == 'MCX'].copy()
    df_bfo = df_all[df_all['exchange'] == 'BFO'].copy()
    
    # 3. Build Base Name Lookup
    base_name_meta_lookup = {}
    df_derivatives = df_all[df_all['instrument_type'].isin(['FUT', 'CE', 'PE'])]
    
    for name, group in df_derivatives.groupby('name'):
        derivatives_exchange = group['exchange'].iloc[0]
        options_data = group[group['instrument_type'].isin(['CE', 'PE'])]
        futures_data = group[group['instrument_type'] == 'FUT']
        
        base_name_meta_lookup[name] = {
            'exchange': derivatives_exchange,
            'options_expiries': sorted(options_data['expiry'].dropna().unique()),
            'futures_expiries': sorted(futures_data['expiry'].dropna().unique())
        }
    
    print(f"[OK] Initialization Complete. Loaded {len(df_all)} instruments.")

except Exception as e:
    print(f"[X] CRITICAL ERROR during data loading: {e}")
    sys.exit(1)

# ==============================================================================
# 3. CORE FUNCTIONS (Copied from your Notebook)
# ==============================================================================

def build_option_chain(symbol, expiry=None, enrich=False):
    # 1. Validate symbol exists
    if symbol not in base_name_meta_lookup:
        print(f"[X] Symbol '{symbol}' not found")
        return {}, pd.DataFrame()
    
    entry = base_name_meta_lookup[symbol]
    exchange = entry['exchange']
    options_expiries = entry['options_expiries']
    
    if not options_expiries:
        return {}, pd.DataFrame()
    
    # 2. Determine Expiry
    if expiry is None:
        expiry = options_expiries[0] # Nearest
    else:
        expiry = pd.to_datetime(expiry)
    
    # 3. Filter DataFrame
    if exchange == 'MCX':
        source_df = df_mcx
    elif exchange == 'BFO':
        source_df = df_bfo
    else:
        source_df = df_all
    
    mask = (
        (source_df['name'] == symbol) &
        (source_df['expiry'] == expiry) &
        (source_df['instrument_type'].isin(['CE', 'PE']))
    )
    chain = source_df[mask].copy()
    
    if chain.empty:
        return {}, pd.DataFrame()
    
    # Optimize
    cols = ['tradingsymbol', 'strike', 'instrument_type', 'expiry', 'lot_size', 'instrument_token', 'tick_size', 'segment']
    chain = chain[[c for c in cols if c in chain.columns]].copy()
    chain = chain.sort_values('strike').reset_index(drop=True)
    
    metadata = {
        'basename': symbol,
        'exchange': exchange,
        'expiry': expiry,
        'lot_size': int(chain['lot_size'].iloc[0]) if 'lot_size' in chain.columns else None
    }
    
    if enrich:
        metadata, chain = enrich_with_market_data(kite, chain, metadata)
        
    return metadata, chain

def enrich_with_market_data(kite_obj, option_chain_df, metadata, chunk_size=250):
    if option_chain_df.empty: return metadata, option_chain_df
    
    exchange = metadata.get('exchange', 'NFO')
    enriched_data = []
    
    # Process in chunks
    tradingsymbols = option_chain_df['tradingsymbol'].tolist()
    symbols_to_quote = [f"{exchange}:{sym}" for sym in tradingsymbols]
    
    all_quotes = {}
    
    # Fetch in batches
    for i in range(0, len(symbols_to_quote), chunk_size):
        batch = symbols_to_quote[i:i+chunk_size]
        try:
            quotes = kite_obj.quote(batch)
            all_quotes.update(quotes)
        except Exception as e:
            print(f"Error fetching quotes: {e}")
            
    # Map back to dataframe
    for idx, row in option_chain_df.iterrows():
        key = f"{exchange}:{row['tradingsymbol']}"
        quote = all_quotes.get(key, {})
        
        depth = quote.get('depth', {})
        buy_depth = depth.get('buy', [])
        sell_depth = depth.get('sell', [])
        
        bid = buy_depth[0]['price'] if buy_depth else 0
        ask = sell_depth[0]['price'] if sell_depth else 0
        ltp = quote.get('last_price', 0)
        
        # Mid price calculation
        if bid > 0 and ask > 0: mid_price = (bid + ask) / 2
        elif bid > 0: mid_price = bid
        elif ask > 0: mid_price = ask
        else: mid_price = ltp
            
        item = row.to_dict()
        item.update({
            'ltp': ltp,
            'mid_price': mid_price,
            'oi': quote.get('oi', 0),
            'volume': quote.get('volume', 0),
            'bid': bid, 'ask': ask
        })
        enriched_data.append(item)
        
    return metadata, pd.DataFrame(enriched_data)

def calculate_iv_vollib(option_chain, metadata=None, risk_free_rate=0.10, oi_filter_pct=0.5):
    if option_chain.empty: return option_chain, 0, 0
    
    # Get Spot Price
    exchange = metadata.get('exchange', 'NFO')
    basename = metadata.get('basename')
    
    # Simple Spot Logic for this app
    try:
        if exchange == 'MCX':
            # Use Future Price as Spot for MCX
            # 1. Find the corresponding future (same expiry or nearest)
            # We need to fetch instrument list again or use a known pattern? 
            # Better approach: Use the underlying name + 'FUT' + Expiry
            # Since we don't have the full instrument list here easily, we rely on a helper or query
            # SIMPLIFICATION: Attempt to find a Future for this symbol expiring this month
            # For robustness in this specific app structure:
            # We will use the 'quote' of the symbol + 'FUT' if possible, or search.
            # actually, let's use the helper lookup!
            
            if basename in base_name_meta_lookup:
               fut_expiries = base_name_meta_lookup[basename].get('futures_expiries', [])
               if fut_expiries:
                   # Use nearest future
                   nearest_fut_expiry = fut_expiries[0]
                   # We need to find the tradingsymbol for this future.
                   # Iterate df_mcx/df_all in global scope?
                   # Since we are inside a function, we rely on 'df_all' being global as per script design
                   
                   # Find future symbol
                   f_mask = (df_all['name'] == basename) & (df_all['instrument_type'] == 'FUT') & (df_all['expiry'] == nearest_fut_expiry)
                   f_row = df_all[f_mask]
                   
                   if not f_row.empty:
                       fut_sym = f"MCX:{f_row.iloc[0]['tradingsymbol']}"
                       ltp_data = kite.ltp(fut_sym)
                       spot_price = ltp_data[fut_sym]['last_price']
                   else:
                       spot_price = option_chain['mid_price'].mean() # Fallback
               else:
                   spot_price = option_chain['mid_price'].mean()
            else:
                spot_price = option_chain['mid_price'].mean()



        elif exchange == 'BFO':
            # BSE Spot
            if basename == 'SENSEX': spot_sym = "BSE:SENSEX"
            elif basename == 'BANKEX': spot_sym = "BSE:BANKEX"
            else: spot_sym = f"BSE:{basename}"
            
            try:
                ltp_data = kite.ltp(spot_sym)
                spot_price = ltp_data[spot_sym]['last_price']
            except:
                 spot_price = option_chain['mid_price'].mean()

        else:
            # NSE Spot
            spot_sym = f"NSE:{basename}" if basename != 'NIFTY' and basename != 'BANKNIFTY' else f"NSE:{basename} 50"
            if basename == 'NIFTY': spot_sym = "NSE:NIFTY 50"
            if basename == 'BANKNIFTY': spot_sym = "NSE:NIFTY BANK"
            if basename == 'MIDCPNIFTY': spot_sym = "NSE:NIFTY MID SELECT"
            if basename == 'FINNIFTY': spot_sym = "NSE:NIFTY FIN SERVICE"
            if basename == 'NIFTYNXT50': spot_sym = "NSE:NIFTY NEXT 50"
            
            ltp_data = kite.ltp(spot_sym)
            spot_price = ltp_data[spot_sym]['last_price']
    except:
        spot_price = 0
        
    if spot_price == 0: return option_chain, 0, 0 # Cannot calc IV without spot
    
    # Time to expiry
    expiry = option_chain['expiry'].iloc[0]
    expiry_dt = expiry + pd.Timedelta(hours=15, minutes=30)
    now = pd.Timestamp.now()
    t = (expiry_dt - now).total_seconds() / (365.25 * 24 * 3600)
    if t <= 0: t = 0.0001
    
    ivs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in option_chain.iterrows():
            try:
                price = float(row['mid_price'])
                strike = float(row['strike'])
                flag = 'c' if row['instrument_type'] == 'CE' else 'p'
                
                if price < 0.05:
                    ivs.append(0)
                    continue
                    
                iv = bsm_iv(price, spot_price, strike, t, risk_free_rate, flag)
                ivs.append(iv * 100)
            except:
                ivs.append(0)
                
    option_chain['iv'] = ivs
    option_chain['iv'] = ivs
    
    # --- ATM IV CALCULATION ---
    atm_iv = 0
    try:
        if spot_price > 0:
            # Find strike closest to spot
            # We use 'strike' column from original or enriched chain
            # Ensure it's numeric
            
            # Filter for non-zero IVs if possible, or just take closest
            # We want the straddle: avg of CE IV and PE IV at ATM strike
            
            # getting row with min distance
            closest_strike_idx = (option_chain['strike'] - spot_price).abs().idxmin()
            atm_strike = option_chain.loc[closest_strike_idx, 'strike']
            
            # get CE and PE IV at this strike
            atm_app_rows = option_chain[option_chain['strike'] == atm_strike]
            
            ce_iv = atm_app_rows[atm_app_rows['instrument_type'] == 'CE']['iv'].max() # max to get value if duplicates
            pe_iv = atm_app_rows[atm_app_rows['instrument_type'] == 'PE']['iv'].max()
            
            # Clean up 0s
            if pd.isna(ce_iv): ce_iv = 0
            if pd.isna(pe_iv): pe_iv = 0
            
            if ce_iv > 0 and pe_iv > 0:
                atm_iv = (ce_iv + pe_iv) / 2
            elif ce_iv > 0:
                atm_iv = ce_iv
            elif pe_iv > 0:
                atm_iv = pe_iv
                
    except Exception as e:
        # keep atm_iv 0
        pass

    return option_chain, spot_price, atm_iv

def show_chain(chain, name):
    # Formats the dataframe into CE | Strike | PE format
    if chain.empty: return pd.DataFrame()
    
    ce_df = chain[chain['instrument_type'] == 'CE'].copy()
    pe_df = chain[chain['instrument_type'] == 'PE'].copy()
    
    # Rename for merge
    ce_df = ce_df.rename(columns={
        'iv': 'CE-IV', 'ltp': 'CE-LTP', 'oi': 'CE-OI',
        'bid': 'CE-BID', 'ask': 'CE-ASK'
    })
    pe_df = pe_df.rename(columns={
        'iv': 'PE-IV', 'ltp': 'PE-LTP', 'oi': 'PE-OI',
        'bid': 'PE-BID', 'ask': 'PE-ASK'
    })
    
    merged = pd.merge(
        ce_df[['strike', 'CE-IV', 'CE-LTP', 'CE-OI', 'CE-BID', 'CE-ASK']], 
        pe_df[['strike', 'PE-IV', 'PE-LTP', 'PE-OI', 'PE-BID', 'PE-ASK']], 
        on='strike', how='outer'
    )
    merged = merged.rename(columns={'strike': 'STRIKE'})
    return merged.sort_values('STRIKE')

def get_option_data(scrip_name, expiry=None):
    # The Master Function called by Dash
    scrip_name = scrip_name.upper().strip()
    
    # 1. Build & Enrich
    meta, chain = build_option_chain(scrip_name, expiry=expiry, enrich=True)
    
    if chain.empty: return pd.DataFrame(), 0, 0
    
    # 2. Calculate IV
    chain, spot_price, atm_iv = calculate_iv_vollib(chain, metadata=meta, oi_filter_pct=0.95)
    
    # 3. Format
    df_formatted = show_chain(chain, scrip_name)
    
    return df_formatted, spot_price, atm_iv

def generate_market_heatmap_data(limit=None):
    # Generates data for IV Heatmap: (Symbol, Moneyness, IV/ATM_IV Ratio)
    heatmap_data = []
    
    # 1. Identify NFO Symbols
    # Use global base_name_meta_lookup
    nfo_symbols = [k for k, v in base_name_meta_lookup.items() if v.get('exchange') == 'NFO']
    
    # Limit for performance if needed (e.g. limit=50)
    if limit:
        nfo_symbols = nfo_symbols[:limit]
        
    print(f"Generating Heatmap for {len(nfo_symbols)} symbols...")
    
    total = len(nfo_symbols)
    for i, sym in enumerate(nfo_symbols):
        # Progress Bar Logic (Simulated)
        print(f"[{i+1}/{total}] Processing {sym}...", end='\r')

        # Add small sleep to reduce load
        time.sleep(0.01)
        
        try:
            # We assume get_option_data is reasonably fast (1 request for quote usually, or cached)
            # Actually get_option_data makes a Kite Quote call. 
            # Doing this 200 times sequentially will take time (e.g. 0.2s * 200 = 40s).
            # We might want to batch this or just let it run.
            
            # Using nearest expiry by default
            try:
                meta, chain = build_option_chain(sym, enrich=True)
            except Exception as e:
                print(f"\n[!] Failed to build chain for {sym}: {e}")
                continue
                
            if chain.empty: continue
            
            try:
                chain, spot_price, atm_iv = calculate_iv_vollib(chain, metadata=meta, oi_filter_pct=0.95)
            except ValueError as e:
                print(f"\n[!] Value Error {sym}: {e}")
                continue
            except Exception as e:
                print(f"\n[!] Calculation Error {sym}: {e}")
                continue
            
            if spot_price <= 0 or atm_iv <= 0: continue
            
            # Calculate Ratios and Moneyness
            # Moneyness = Strike / Spot
            # Ratio = IV / ATM_IV
            
            # Filter relevant strikes (e.g. 0.8 to 1.2 moneyness) to keep heatmap focused
            # OR just take all.
            
            for _, row in chain.iterrows():
                strike = row['strike']
                iv = row['iv']
                
                if iv <= 0: continue
                
                moneyness = round(strike / spot_price, 2) # Round to 2 decimals for binning
                ratio = iv / atm_iv
                
                spread = abs(row['ask'] - row['bid']) if row['ask'] > 0 and row['bid'] > 0 else 0
                oi = row['oi']
                
                heatmap_data.append({
                    'Symbol': sym,
                    'Moneyness': moneyness,
                    'Ratio': ratio,
                    'IV': iv,
                    'Spread': spread,
                    'OI': oi,
                    'Type': row['instrument_type']
                })
                
        except Exception as e:
            print(f"\n[!] Error processing {sym}: {e}")
            continue
            
    return pd.DataFrame(heatmap_data)

# --- DISK CACHE for Heatmap Data ---
# Parquet-based cache with 5-minute TTL (memory-efficient)
_CACHE_DIR = os.path.join(tempfile.gettempdir(), 'kite_heatmap_cache')
os.makedirs(_CACHE_DIR, exist_ok=True)
_CACHE_FILE = os.path.join(_CACHE_DIR, 'heatmap_data.parquet')
_CACHE_TTL_SECONDS = 300  # 5 minutes

def _get_heatmap_data(limit=100, force_refresh=False):
    """Read heatmap data from disk cache if fresh, otherwise generate and save."""
    if not force_refresh and os.path.exists(_CACHE_FILE):
        file_age = time.time() - os.path.getmtime(_CACHE_FILE)
        if file_age < _CACHE_TTL_SECONDS:
            print(f"Cache HIT: Loading from disk (age: {file_age:.0f}s / {_CACHE_TTL_SECONDS}s)")
            return pd.read_parquet(_CACHE_FILE)
        else:
            print(f"Cache EXPIRED: {file_age:.0f}s > {_CACHE_TTL_SECONDS}s TTL")
    else:
        print("Cache MISS: No cache file found, generating fresh data...")
    
    df = generate_market_heatmap_data(limit=limit)
    if not df.empty:
        df.to_parquet(_CACHE_FILE, index=False)
        print(f"Cache SAVED: {len(df)} rows -> {_CACHE_FILE}")
    return df


# ==============================================================================
# 4. DASH WEB APP CONFIGURATION
# ==============================================================================
# ==============================================================================
# 4. DASH WEB APP CONFIGURATION
# ==============================================================================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Kite Option Smile"

# --- SIDEBAR STYLE ---
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "backgroundColor": "#f8f9fa",
}

# --- CONTENT STYLE ---
CONTENT_STYLE = {
    "marginLeft": "18rem",
    "marginRight": "2rem",
    "padding": "2rem 1rem",
}

# --- LAYOUTS ---

def layout_market_overview():
    curr_user = "Unknown"
    
    if 'kite' in globals():
        try:
            profile = kite.profile()
            curr_user = profile.get('user_name', 'Unknown')
        except:
             pass

    return html.Div([
        # Welcome Card
        html.Div([
            html.H2(f"Welcome, {curr_user}", style={'color': '#007bff'}),
            html.P("Select 'Volatility Analyzer' from the sidebar to start analyzing individual scrips."),
        ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
        
        # Heatmap Section
        html.Div([
            html.H3("Market-Wide IV Skew Heatmap", style={'textAlign': 'center'}),
            html.P("Visualizing the IV Skew (IV / ATM IV) across NFO symbols.", style={'textAlign': 'center', 'color': '#666'}),
            
            html.Div([
                html.Button("Generate Heatmap (All NFO)", id="btn-heatmap", n_clicks=0, 
                            style={'padding': '10px 20px', 'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 'cursor': 'pointer', 'fontSize': '16px'}),
                html.Span("  (Note: This may take a minute)", style={'fontStyle': 'italic', 'fontSize': '12px', 'marginLeft': '10px'})
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            dcc.Loading(
                id="loading-heatmap",
                children=[dcc.Graph(id="graph-heatmap", style={'height': '800px'})],
                type="cube"
            )
        ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'})
    ])

def layout_pcr_view():
    return html.Div([
        # Header Card
        html.Div([
            html.H2("PCR Skew Heatmap", style={'color': '#007bff'}),
            html.P("Visualizing Put-Call OI Ratio (PE OI / CE OI) across NFO symbols by moneyness.", style={'color': '#666'}),
        ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
        
        # Heatmap Section
        html.Div([
            html.Div([
                html.Button("Generate PCR Heatmap", id="btn-pcr-heatmap", n_clicks=0, 
                            style={'padding': '10px 20px', 'backgroundColor': '#17a2b8', 'color': 'white', 'border': 'none', 'cursor': 'pointer', 'fontSize': '16px'}),
                html.Span("  (Uses cached data from IV Heatmap if available)", style={'fontStyle': 'italic', 'fontSize': '12px', 'marginLeft': '10px'})
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            dcc.Loading(
                id="loading-pcr-heatmap",
                children=[dcc.Graph(id="graph-pcr-heatmap", style={'height': '800px'})],
                type="cube"
            )
        ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'})
    ])

def layout_iv_ratio_view():
    return html.Div([
        # Header Card
        html.Div([
            html.H2("Put/Call IV Ratio Heatmap", style={'color': '#007bff'}),
            html.P("Visualizing Put IV / Call IV ratio across NFO symbols by moneyness.", style={'color': '#666'}),
        ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),
        
        # Heatmap Section
        html.Div([
            html.Div([
                html.Button("Generate IV Ratio Heatmap", id="btn-ivratio-heatmap", n_clicks=0, 
                            style={'padding': '10px 20px', 'backgroundColor': '#6f42c1', 'color': 'white', 'border': 'none', 'cursor': 'pointer', 'fontSize': '16px'}),
                html.Span("  (Uses cached data from IV Heatmap if available)", style={'fontStyle': 'italic', 'fontSize': '12px', 'marginLeft': '10px'})
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            dcc.Loading(
                id="loading-ivratio-heatmap",
                children=[dcc.Graph(id="graph-ivratio-heatmap", style={'height': '800px'})],
                type="cube"
            )
        ], style={'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'})
    ])

layout_analyzer = html.Div(children=[
    html.H1("Option Volatility Smile Analyzer", style={'textAlign': 'center', 'color': '#333'}),
    
    html.Div([
        html.Label("Scrip Name:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Input(
            id='scrip-input', 
            type='text', 
            value='NIFTY', 
            placeholder='e.g. NIFTY, GOLDM',
            style={'padding': '10px', 'fontSize': '16px', 'width': '200px'}
        ),
        html.Button(
            'Analyze', 
            id='submit-btn', 
            n_clicks=0,
            style={'padding': '10px 20px', 'fontSize': '16px', 'marginLeft': '10px', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'cursor': 'pointer'}
        ),
        
        html.Div([
            html.Label("Expiry:", style={'fontWeight': 'bold', 'marginRight': '10px', 'marginLeft': '20px'}),
            dcc.Dropdown(
                id='expiry-dropdown',
                style={'width': '200px', 'textAlign': 'left'}
            )
        ], style={'display': 'inline-block', 'verticalAlign': 'middle'})
        
    ], style={'textAlign': 'center', 'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'}),
    
    dcc.Loading(
        type="dot",
        children=[
            dcc.Graph(id='smile-graph', style={'height': '600px'}),
            html.Div(id='click-data-output', style={'padding': '20px', 'textAlign': 'center'}),
            html.Div(id='error-msg', style={'color': 'red', 'textAlign': 'center', 'marginTop': '20px'})
        ]
    )
])

# --- MAIN LAYOUT ---
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    
    # Persistent Stores for figure caching across page navigations
    dcc.Store(id='store-heatmap-fig', storage_type='session'),
    dcc.Store(id='store-pcr-fig', storage_type='session'),
    dcc.Store(id='store-ivratio-fig', storage_type='session'),
    
    # Sidebar
    html.Div([
        html.H3("Kite Analytics", className="display-4"),
        html.Hr(),
        html.Plaintext("Short Menu"),
        
        dcc.Link('Profile / Home', href='/', style={'display': 'block', 'padding': '10px', 'textDecoration': 'none', 'color': '#333', 'fontWeight': 'bold'}),
        dcc.Link('Volatility Analyzer', href='/analyzer', style={'display': 'block', 'padding': '10px', 'textDecoration': 'none', 'color': '#333', 'fontWeight': 'bold'}),
        dcc.Link('PCR Skew', href='/pcr', style={'display': 'block', 'padding': '10px', 'textDecoration': 'none', 'color': '#333', 'fontWeight': 'bold'}),
        dcc.Link('IV Ratio', href='/iv-ratio', style={'display': 'block', 'padding': '10px', 'textDecoration': 'none', 'color': '#333', 'fontWeight': 'bold'}),
        
    ], style=SIDEBAR_STYLE),
    
    # Content
    html.Div(id="page-content", style=CONTENT_STYLE)
])


# --- ROUTING CALLBACK ---
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/" or pathname == "/home":
        return layout_market_overview()
    elif pathname == "/analyzer":
        return layout_analyzer
    elif pathname == "/pcr":
        return layout_pcr_view()
    elif pathname == "/iv-ratio":
        return layout_iv_ratio_view()
    else:
        return html.Div(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ],
            className="p-3 bg-light rounded-3",
        )

@app.callback(
    [Output('smile-graph', 'figure'), 
     Output('error-msg', 'children'),
     Output('expiry-dropdown', 'options'),
     Output('expiry-dropdown', 'value')],
    [Input('submit-btn', 'n_clicks'),
     Input('expiry-dropdown', 'value')],
    [State('scrip-input', 'value'),
     State('expiry-dropdown', 'options')]
)
def update_dashboard(n_clicks, expiry_value, scrip_name, existing_options):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'submit-btn'
    
    if not scrip_name:
        return go.Figure(), "", [], None
    
    scrip_name = scrip_name.upper().strip()
    
    # Initialize variables
    final_expiry = None
    final_options = existing_options if existing_options else []
    
    # LOGIC: Re-fetch expiries if Button Clicked or Options Empty
    if trigger_id == 'submit-btn' or not final_options:
        # Check if symbol exists logic
        if scrip_name in base_name_meta_lookup:
            exps = base_name_meta_lookup[scrip_name].get('options_expiries', [])
            if exps:
                # Create dropdown options
                final_options = [{'label': str(pd.to_datetime(d).date()), 'value': str(d)} for d in exps]
                final_expiry = final_options[0]['value'] # Default to nearest
            else:
                final_options = []
                final_expiry = None
        else:
            final_options = []
            final_expiry = None
            
    elif trigger_id == 'expiry-dropdown':
        final_expiry = expiry_value
        
    # Prevent logic run if we switched scrip but have no expiry
    if not final_expiry and final_options:
         final_expiry = final_options[0]['value']

    try:
        # RUN THE LOGIC with selected expiry
        df, spot_price, atm_iv = get_option_data(scrip_name, expiry=final_expiry)
        
        if df.empty:
            return go.Figure(), f"No data found for {scrip_name}. Check spelling or market status.", final_options, final_expiry
        
        # Plotting
        fig = go.Figure()

        # --- BACKGROUND OI BARS (Secondary Axis) ---
        # CE OI (Green, Transparent)
        fig.add_trace(go.Bar(
            x=df['STRIKE'], y=df['CE-OI'],
            name='CE OI',
            marker=dict(color='rgba(40, 167, 69, 0.2)'), # Green with 0.2 opacity
            yaxis='y2'
        ))
        
        # PE OI (Red, Transparent)
        fig.add_trace(go.Bar(
            x=df['STRIKE'], y=df['PE-OI'],
            name='PE OI',
            marker=dict(color='rgba(220, 53, 69, 0.2)'), # Red with 0.2 opacity
            yaxis='y2'
        ))

        # --- FOREGROUND IV LINES (Primary Axis) ---
        
        # CE IV Line (Green)
        fig.add_trace(go.Scatter(
            x=df['STRIKE'], y=df['CE-IV'],
            mode='lines+markers', name='CE IV',
            line=dict(color='#28a745', width=3),
            marker=dict(size=6),
            customdata=df[['CE-LTP', 'CE-BID', 'CE-ASK', 'CE-OI']].values,
            hovertemplate="<b>Strike: %{x}</b><br>IV: %{y:.2f}%<br>LTP: %{customdata[0]}<br>Bid: %{customdata[1]}<br>Ask: %{customdata[2]}<br>OI: %{customdata[3]}<extra></extra>"
        ))
        
        # PE IV Line (Red)
        fig.add_trace(go.Scatter(
            x=df['STRIKE'], y=df['PE-IV'],
            mode='lines+markers', name='PE IV',
            line=dict(color='#dc3545', width=3),
            marker=dict(size=6),
            customdata=df[['PE-LTP', 'PE-BID', 'PE-ASK', 'PE-OI']].values,
            hovertemplate="<b>Strike: %{x}</b><br>IV: %{y:.2f}%<br>LTP: %{customdata[0]}<br>Bid: %{customdata[1]}<br>Ask: %{customdata[2]}<br>OI: %{customdata[3]}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Volatility Smile: {scrip_name}",
            xaxis_title="Strike Price",
            yaxis_title="Implied Volatility (%)",
            
            # Secondary Y-Axis for OI
            yaxis2=dict(
                title="Open Interest",
                overlaying="y",
                side="right",
                showgrid=False, # cleaner look
                range=[0, df[['CE-OI', 'PE-OI']].max().max() * 3] # Scale it down so bars stay low
            ),
            
            template="plotly_white",
            hovermode="x unified",

            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        # Add Spot Price Line
        if spot_price > 0:
            anno_text = f"Spot: {spot_price}"
            if atm_iv > 0:
                anno_text += f" | ATM IV: {atm_iv:.2f}%"
                
            fig.add_vline(
                x=spot_price, 
                line_width=2, 
                line_dash="dash", 
                line_color="black",
                annotation_text=anno_text, 
                annotation_position="top right"
            )
        
        return fig, "", final_options, final_expiry
        
    except Exception as e:
        return go.Figure(), f"An error occurred: {str(e)}", final_options, final_expiry

@app.callback(
    [Output("graph-heatmap", "figure"),
     Output("store-heatmap-fig", "data")],
    [Input("btn-heatmap", "n_clicks")],
    [State("store-heatmap-fig", "data")]
)
def update_heatmap(n_clicks, stored_fig):
    # On page load (no click), restore from Store if available
    if not n_clicks:
        if stored_fig:
            return go.Figure(stored_fig), dash.no_update
        return go.Figure(), dash.no_update
        
    # Generate Data (uses disk cache with 5-min TTL)
    df_heat = _get_heatmap_data(limit=100)
    
    if df_heat.empty:
        return go.Figure(), dash.no_update
        
    # Bin Moneyness - Granularity 0.02
    bin_step = 0.02
    df_heat['Moneyness_Bin'] = (df_heat['Moneyness'] / bin_step).round() * bin_step
    
    # Define Color Scale (Standard Ratio)
    z_mid = 1.0
    colors = 'RdBu_r'
    title = "IV / ATM IV"

    # Helper to pivot multiple metrics aligned
    def create_pivot(df_sub):
        # We need to ensure all pivots have same shape. 
        # Pivot each metric
        p_ratio = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='Ratio', aggfunc='mean').sort_index(ascending=False)
        p_oi = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='OI', aggfunc='mean').sort_index(ascending=False)
        p_spread = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='Spread', aggfunc='mean').sort_index(ascending=False)
        
        # Filter columns to 0.8-1.2 (using ratio columns as master)
        cols = [c for c in p_ratio.columns if 0.8 <= c <= 1.2]
        
        return p_ratio[cols], p_oi[cols], p_spread[cols]

    # --- CE HEATMAP ---
    df_ce = df_heat[df_heat['Type'] == 'CE']
    ce_ratio, ce_oi, ce_spread = create_pivot(df_ce)
    
    # Custom Data for CE: Stack along last axis
    # shape: (rows, cols, 2) -> index 0: OI, index 1: Spread
    import numpy as np
    ce_custom = np.dstack((ce_oi.values, ce_spread.values))
    
    # --- PE HEATMAP ---
    df_pe = df_heat[df_heat['Type'] == 'PE']
    pe_ratio, pe_oi, pe_spread = create_pivot(df_pe)
    
    # Custom Data for PE
    pe_custom = np.dstack((pe_oi.values, pe_spread.values))
    
    # Create Subplots: 2 Rows
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"CE {title}", f"PE {title}")
    )
    
    # Common Heatmap Args
    hm_args = dict(
        colorscale=colors,
        zmid=z_mid,
        colorbar=dict(title=title, len=0.45)
    )

    # Add CE Trace
    fig.add_trace(go.Heatmap(
        z=ce_ratio.values,
        x=ce_ratio.columns,
        y=ce_ratio.index,
        customdata=ce_custom,
        colorbar_y=0.8,
        hovertemplate="<b>Symbol: %{y}</b><br>Moneyness: %{x}<br>Ratio: %{z:.2f}<br>OI: %{customdata[0]:,}<br>Spread: %{customdata[1]:.2f}<extra>CE</extra>",
        **hm_args
    ), row=1, col=1)
    
    # Add PE Trace
    fig.add_trace(go.Heatmap(
        z=pe_ratio.values,
        x=pe_ratio.columns,
        y=pe_ratio.index,
        customdata=pe_custom,
        colorbar_y=0.2,
        hovertemplate="<b>Symbol: %{y}</b><br>Moneyness: %{x}<br>Ratio: %{z:.2f}<br>OI: %{customdata[0]:,}<br>Spread: %{customdata[1]:.2f}<extra>PE</extra>",
        **hm_args
    ), row=2, col=1)
    
    # Layout
    fig.update_layout(
        title=f"NFO {title} Heatmap (All {len(ce_ratio)} Symbols)",
        height=max(800, len(ce_ratio) * 25), 
        xaxis2={'title': "Moneyness (Strike / Spot)", 'tickmode': 'linear', 'dtick': 0.05}
    )
    
    return fig, fig

@app.callback(
    Output('click-data-output', 'children'),
    Input('smile-graph', 'clickData')
)
def display_click_data(clickData):
    if not clickData:
        return html.Div("Click on a data point on the graph to see details.", style={'color': '#555', 'fontStyle': 'italic'})
    
    try:
        point = clickData['points'][0]
        strike = point['x']
        iv = point['y']
        
        # customdata = [LTP, BID, ASK, OI]
        custom_data = point.get('customdata', [0, 0, 0, 0])
        ltp, bid, ask, oi = custom_data[0], custom_data[1], custom_data[2], custom_data[3]
        
        # Simple Card Style
        return html.Div([
            html.H3(f"Selected Strike: {strike}", style={'margin': '0 0 10px 0', 'color': '#333'}),
            html.Div([
                html.Div([
                    html.Strong("IV:"), html.Span(f" {iv:.2f}%")
                ], style={'display': 'inline-block', 'margin': '0 10px'}),
                html.Div([
                    html.Strong("LTP:"), html.Span(f" {ltp}")
                ], style={'display': 'inline-block', 'margin': '0 10px'}),
                html.Div([
                    html.Strong("Bid:"), html.Span(f" {bid}")
                ], style={'display': 'inline-block', 'margin': '0 10px'}),
                 html.Div([
                    html.Strong("Ask:"), html.Span(f" {ask}")
                ], style={'display': 'inline-block', 'margin': '0 10px'}),
                html.Div([
                    html.Strong("OI:"), html.Span(f" {oi}")
                ], style={'display': 'inline-block', 'margin': '0 10px'}),
            ], style={'backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '8px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'display': 'inline-block'})
        ])
    except Exception as e:
        return html.Div(f"Error displaying details: {str(e)}")

# --- PCR HEATMAP CALLBACK ---
@app.callback(
    [Output("graph-pcr-heatmap", "figure"),
     Output("store-pcr-fig", "data")],
    [Input("btn-pcr-heatmap", "n_clicks")],
    [State("store-pcr-fig", "data")]
)
def update_pcr_heatmap(n_clicks, stored_fig):
    # On page load (no click), restore from Store if available
    if not n_clicks:
        if stored_fig:
            return go.Figure(stored_fig), dash.no_update
        return go.Figure(), dash.no_update
    
    # Use disk cache
    df_heat = _get_heatmap_data(limit=100)
    
    if df_heat.empty:
        return go.Figure(), dash.no_update
    
    # Bin Moneyness - same granularity as IV heatmap
    bin_step = 0.02
    df_heat = df_heat.copy()
    df_heat['Moneyness_Bin'] = (df_heat['Moneyness'] / bin_step).round() * bin_step
    
    # Filter to 0.8-1.2 range
    df_heat = df_heat[(df_heat['Moneyness_Bin'] >= 0.8) & (df_heat['Moneyness_Bin'] <= 1.2)]
    
    # Pivot CE OI and PE OI separately
    df_ce = df_heat[df_heat['Type'] == 'CE']
    df_pe = df_heat[df_heat['Type'] == 'PE']
    
    pivot_ce_oi = df_ce.pivot_table(index='Symbol', columns='Moneyness_Bin', values='OI', aggfunc='sum').sort_index(ascending=False)
    pivot_pe_oi = df_pe.pivot_table(index='Symbol', columns='Moneyness_Bin', values='OI', aggfunc='sum').sort_index(ascending=False)
    
    # Align to same index and columns
    all_symbols = sorted(set(pivot_ce_oi.index) | set(pivot_pe_oi.index), reverse=True)
    all_bins = sorted(set(list(pivot_ce_oi.columns) + list(pivot_pe_oi.columns)))
    
    pivot_ce_oi = pivot_ce_oi.reindex(index=all_symbols, columns=all_bins, fill_value=0)
    pivot_pe_oi = pivot_pe_oi.reindex(index=all_symbols, columns=all_bins, fill_value=0)
    
    # PCR = PE OI / CE OI (avoid div by zero)
    pcr = pivot_pe_oi / pivot_ce_oi.replace(0, np.nan)
    
    # Build customdata: stack CE OI and PE OI for hover
    custom = np.dstack((pivot_ce_oi.values, pivot_pe_oi.values))
    
    fig = go.Figure(go.Heatmap(
        z=pcr.values,
        x=pcr.columns,
        y=pcr.index,
        customdata=custom,
        colorscale='RdBu',
        zmid=1.0,
        zmin=0,
        zmax=3.0,
        colorbar=dict(title="PCR"),
        hovertemplate="<b>Symbol: %{y}</b><br>Moneyness: %{x}<br>PCR: %{z:.2f}<br>CE OI: %{customdata[0]:,}<br>PE OI: %{customdata[1]:,}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"NFO Put-Call Ratio Heatmap ({len(all_symbols)} Symbols)",
        height=max(800, len(all_symbols) * 25),
        xaxis={'title': 'Moneyness (Strike / Spot)', 'tickmode': 'linear', 'dtick': 0.05}
    )
    
    return fig, fig

# --- PUT/CALL IV RATIO HEATMAP CALLBACK ---
@app.callback(
    [Output("graph-ivratio-heatmap", "figure"),
     Output("store-ivratio-fig", "data")],
    [Input("btn-ivratio-heatmap", "n_clicks")],
    [State("store-ivratio-fig", "data")]
)
def update_ivratio_heatmap(n_clicks, stored_fig):
    # On page load (no click), restore from Store if available
    if not n_clicks:
        if stored_fig:
            return go.Figure(stored_fig), dash.no_update
        return go.Figure(), dash.no_update
    
    # Use disk cache
    df_heat = _get_heatmap_data(limit=100)
    
    if df_heat.empty:
        return go.Figure(), dash.no_update
    
    # Bin Moneyness
    bin_step = 0.02
    df_heat = df_heat.copy()
    df_heat['Moneyness_Bin'] = (df_heat['Moneyness'] / bin_step).round() * bin_step
    
    # Filter to 0.8-1.2 range
    df_heat = df_heat[(df_heat['Moneyness_Bin'] >= 0.8) & (df_heat['Moneyness_Bin'] <= 1.2)]
    
    # Pivot CE IV and PE IV separately
    df_ce = df_heat[df_heat['Type'] == 'CE']
    df_pe = df_heat[df_heat['Type'] == 'PE']
    
    pivot_ce_iv = df_ce.pivot_table(index='Symbol', columns='Moneyness_Bin', values='IV', aggfunc='mean').sort_index(ascending=False)
    pivot_pe_iv = df_pe.pivot_table(index='Symbol', columns='Moneyness_Bin', values='IV', aggfunc='mean').sort_index(ascending=False)
    
    # Align to same index and columns
    all_symbols = sorted(set(pivot_ce_iv.index) | set(pivot_pe_iv.index), reverse=True)
    all_bins = sorted(set(list(pivot_ce_iv.columns) + list(pivot_pe_iv.columns)))
    
    pivot_ce_iv = pivot_ce_iv.reindex(index=all_symbols, columns=all_bins)
    pivot_pe_iv = pivot_pe_iv.reindex(index=all_symbols, columns=all_bins)
    
    # IV Ratio = PE IV / CE IV (avoid div by zero)
    iv_ratio = pivot_pe_iv / pivot_ce_iv.replace(0, np.nan)
    
    # Build customdata: stack CE IV and PE IV for hover
    custom = np.dstack((pivot_ce_iv.values, pivot_pe_iv.values))
    
    fig = go.Figure(go.Heatmap(
        z=iv_ratio.values,
        x=iv_ratio.columns,
        y=iv_ratio.index,
        customdata=custom,
        colorscale='RdBu',
        zmid=1.0,
        zmin=0.5,
        zmax=1.5,
        colorbar=dict(title="PE/CE IV"),
        hovertemplate="<b>Symbol: %{y}</b><br>Moneyness: %{x}<br>PE/CE IV: %{z:.2f}<br>CE IV: %{customdata[0]:.2f}%<br>PE IV: %{customdata[1]:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"NFO Put/Call IV Ratio Heatmap ({len(all_symbols)} Symbols)",
        height=max(800, len(all_symbols) * 25),
        xaxis={'title': 'Moneyness (Strike / Spot)', 'tickmode': 'linear', 'dtick': 0.05}
    )
    
    return fig, fig

# ==============================================================================
# 5. SERVER RUN
# ==============================================================================
if __name__ == '__main__':
    # debug=False is safer when using global variables in complex scripts
    app.run(debug=True, use_reloader=False)