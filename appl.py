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
from dash import Dash, html, dcc, Input, Output, State, dash_table
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
    if t <= 0: t = 1 / 365.25  # Min 1 day to avoid IV explosion on expiry day
    
    ivs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, row in option_chain.iterrows():
            try:
                price = float(row['mid_price'])
                strike = float(row['strike'])
                bid = float(row.get('bid', 0))
                ask = float(row.get('ask', 0))
                oi = float(row.get('oi', 0))
                flag = 'c' if row['instrument_type'] == 'CE' else 'p'
                
                # Guard 1: Skip if strike is invalid
                if strike <= 0:
                    ivs.append(0)
                    continue
                
                # Guard 2: Skip if no real market (bid or ask is 0)
                if bid <= 0 or ask <= 0:
                    ivs.append(0)
                    continue
                
                # Guard 3: Skip if price is too low to extract meaningful IV
                if price < 0.05:
                    ivs.append(0)
                    continue
                
                # Guard 4: Intrinsic value floor — skip if no extrinsic value
                if flag == 'c':
                    intrinsic = max(0, spot_price - strike)
                else:
                    intrinsic = max(0, strike - spot_price)
                
                if price <= intrinsic + 0.01:
                    # Mid-price is at or below intrinsic — no time value to extract IV from
                    ivs.append(0)
                    continue
                
                # Calculate IV
                iv = bsm_iv(price, spot_price, strike, t, risk_free_rate, flag)
                iv_pct = iv * 100
                
                # Guard 5: Clamp to sane range (0-500%)
                if iv_pct < 0 or iv_pct > 500:
                    ivs.append(0)
                    continue
                
                ivs.append(iv_pct)
                
            except ValueError:
                # bsm_iv can raise ValueError for impossible prices
                ivs.append(0)
            except Exception as e:
                ivs.append(0)
                
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
    merged = merged.sort_values('STRIKE')
    
    # Replace IV=0 with NaN so Plotly draws gaps instead of dips to zero
    merged['CE-IV'] = merged['CE-IV'].replace(0, np.nan)
    merged['PE-IV'] = merged['PE-IV'].replace(0, np.nan)
    
    return merged

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

def generate_market_heatmap_data(limit=150):
    # Generates data for IV Heatmap + Calendar Spread
    # Covers ALL exchanges: NFO, MCX, BFO
    # Fetches BOTH nearest (Rank 1) and next-month (Rank 2) expiry per symbol
    heatmap_data = []
    
    # Collect symbols from all exchanges
    all_symbols = []
    for k, v in base_name_meta_lookup.items():
        exchange = v.get('exchange', '')
        if exchange in ('NFO', 'MCX', 'BFO'):
            all_symbols.append((k, exchange))
    
    # Limit for performance if needed during testing
    if limit:
        all_symbols = all_symbols[:limit]
        
    print(f"Generating Heatmap for {len(all_symbols)} symbols (NFO + MCX + BFO), 2 expiries each...")
    
    total = len(all_symbols)
    for i, (sym, exchange) in enumerate(all_symbols):
        # Progress Bar Logic
        print(f"[{i+1}/{total}] Processing {sym} ({exchange})...", end='\r')

        # Add small sleep to reduce load
        time.sleep(0.01)
        
        try:
            entry = base_name_meta_lookup.get(sym, {})
            expiries = entry.get('options_expiries', [])
            
            if not expiries:
                continue
            
            # Process up to 2 expiries: nearest and next-month
            expiries_to_process = expiries[:2]  # [0]=nearest, [1]=next month (if exists)
            
            for exp_rank, exp_date in enumerate(expiries_to_process, start=1):
                try:
                    meta, chain = build_option_chain(sym, expiry=exp_date, enrich=True)
                except Exception as e:
                    print(f"\n[!] Failed to build chain for {sym} exp={exp_date}: {e}")
                    continue
                    
                if chain.empty: continue
                
                try:
                    chain, spot_price, atm_iv = calculate_iv_vollib(chain, metadata=meta, oi_filter_pct=0.95)
                except ValueError as e:
                    print(f"\n[!] Value Error {sym} exp={exp_date}: {e}")
                    continue
                except Exception as e:
                    print(f"\n[!] Calculation Error {sym} exp={exp_date}: {e}")
                    continue
                
                if spot_price <= 0 or atm_iv <= 0: continue
                
                for _, row in chain.iterrows():
                    strike = row['strike']
                    iv = row['iv']
                    
                    if iv <= 0: continue
                    
                    moneyness = round(strike / spot_price, 2)
                    ratio = iv / atm_iv
                    
                    spread = abs(row['ask'] - row['bid']) if row['ask'] > 0 and row['bid'] > 0 else 0
                    oi = row['oi']
                    
                    heatmap_data.append({
                        'Symbol': sym,
                        'Exchange': exchange,
                        'Expiry_Rank': exp_rank,
                        'Expiry': str(exp_date.date()) if hasattr(exp_date, 'date') else str(exp_date),
                        'Strike': strike,
                        'Spot': spot_price,
                        'Moneyness': moneyness,
                        'Ratio': ratio,
                        'IV': iv,
                        'ATM_IV': atm_iv,
                        'Bid': row['bid'],
                        'Ask': row['ask'],
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

def _get_heatmap_data(limit=None, force_refresh=False):
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

def compute_max_pain(df_heat):
    """Compute Max Pain for each symbol from cached heatmap data.
    Returns summary DataFrame and per-symbol pain profiles."""
    results = []
    pain_profiles = {}  # {symbol: DataFrame with Strike, CE_Pain, PE_Pain, Total_Pain}
    
    for sym, grp in df_heat.groupby('Symbol'):
        spot = grp['Spot'].iloc[0]
        strikes = sorted(grp['Strike'].unique())
        
        # Get CE and PE OI per strike
        ce_data = grp[grp['Type'] == 'CE'][['Strike', 'OI']].groupby('Strike')['OI'].sum()
        pe_data = grp[grp['Type'] == 'PE'][['Strike', 'OI']].groupby('Strike')['OI'].sum()
        
        pain_at_strike = []
        for K in strikes:
            # If price settles at K:
            # CE holders gain max(0, K - strike) for strikes < K -> writers lose
            # PE holders gain max(0, strike - K) for strikes > K -> writers lose
            ce_pain = sum(ce_data.get(s, 0) * max(0, K - s) for s in ce_data.index)
            pe_pain = sum(pe_data.get(s, 0) * max(0, s - K) for s in pe_data.index)
            total = ce_pain + pe_pain
            pain_at_strike.append({
                'Strike': K, 'CE_Pain': ce_pain, 'PE_Pain': pe_pain,
                'Total_Pain': total,
                'CE_OI': ce_data.get(K, 0), 'PE_OI': pe_data.get(K, 0)
            })
        
        pdf = pd.DataFrame(pain_at_strike)
        pain_profiles[sym] = pdf
        
        if pdf.empty:
            continue
        
        # Max Pain = strike with minimum Total Pain
        mp_row = pdf.loc[pdf['Total_Pain'].idxmin()]
        max_pain_strike = mp_row['Strike']
        distance_pct = round((max_pain_strike - spot) / spot * 100, 2)
        
        # Top 5 highest-pain strikes
        top5 = pdf.nlargest(5, 'Total_Pain')
        
        results.append({
            'Symbol': sym,
            'Spot': round(spot, 2),
            'Max Pain': round(max_pain_strike, 2),
            'Distance%': distance_pct,
            'Bias': '🟢 Bull' if distance_pct > 0 else '🔴 Bear',
            'Top5_Strikes': top5['Strike'].tolist(),
            'Top5_Pain': top5['Total_Pain'].tolist()
        })
    
    summary = pd.DataFrame(results).sort_values('Symbol')
    return summary, pain_profiles


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

# Max Pain View
def layout_max_pain_view():
    return html.Div([
        html.H2("Max Pain Analysis", style={'textAlign': 'center', 'color': '#333'}),

        html.Button("Generate Max Pain", id="btn-maxpain", n_clicks=0,
                     style={'display': 'block', 'margin': '20px auto', 'padding': '10px 20px',
                            'fontSize': '16px', 'backgroundColor': '#6f42c1', 'color': 'white',
                            'border': 'none', 'cursor': 'pointer', 'borderRadius': '5px'}),

        dcc.Loading([
            # Section 1: Summary Table
            html.Div(id='maxpain-table-container', style={'margin': '20px 0'}),

            # Section 2: Top 5 Pain Heatmap
            dcc.Graph(id='graph-maxpain-heatmap', style={'height': '900px'}),

            # Section 3: Drill-down
            html.H3(id='drilldown-title', style={'textAlign': 'center', 'marginTop': '30px', 'color': '#555'}),
            dcc.Graph(id='graph-maxpain-drilldown', style={'height': '500px'}),
        ]),

        # Section 4: Interpretation Notes
        html.Div([
            html.H4("How to Interpret Max Pain"),
            html.Ul([
                html.Li("Max Pain = strike where total option buyer losses are maximized (option writers profit most)."),
                html.Li("If Spot > Max Pain → Bearish bias (market may pull back toward Max Pain)."),
                html.Li("If Spot < Max Pain → Bullish bias (market may rally toward Max Pain)."),
                html.Li("High pain concentration at one strike = strong magnet for expiry settlement."),
                html.Li("Click a row in the table to see the full pain profile for that symbol."),
            ])
        ], style={'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '8px', 'marginTop': '30px'})
    ])

# Calendar IV Spread View
def layout_calendar_view():
    return html.Div([
        html.H2("Calendar IV Spread", style={'textAlign': 'center', 'color': '#333'}),
        html.P("Near-month IV vs Next-month IV — find calendar spread opportunities", style={'textAlign': 'center', 'color': '#666'}),

        html.Button("Generate Calendar Spread", id="btn-calendar", n_clicks=0,
                     style={'display': 'block', 'margin': '20px auto', 'padding': '10px 20px',
                            'fontSize': '16px', 'backgroundColor': '#fd7e14', 'color': 'white',
                            'border': 'none', 'cursor': 'pointer', 'borderRadius': '5px'}),

        dcc.Loading([
            # Section 1: ATM Calendar Spread Bar Chart
            dcc.Graph(id='graph-calendar-atm', style={'height': '900px'}),

            # Section 2: Summary DataTable
            html.Div(id='calendar-table-container', style={'margin': '20px 0'}),

            # Section 3: Strike-Level Drill-Down
            html.H3(id='calendar-drilldown-title', style={'textAlign': 'center', 'marginTop': '30px', 'color': '#555'}),
            dcc.Graph(id='graph-calendar-drilldown', style={'height': '550px'}),
        ]),

        # Store for calendar data
        dcc.Store(id='store-calendar-data', storage_type='session'),

        # Interpretation
        html.Div([
            html.H4("How to Read Calendar Spreads"),
            html.Ul([
                html.Li("Positive spread (Near IV > Far IV) = Contango (normal). Sell near, buy far."),
                html.Li("Negative spread (Near IV < Far IV) = Backwardation (unusual). Buy near, sell far — or investigate."),
                html.Li("Large |spread| = potential calendar spread trade opportunity."),
                html.Li("Click a row in the table to see the full strike-by-strike IV comparison."),
                html.Li("Green bars = Far IV higher at that strike (backwardation). Red = Near IV higher (contango)."),
            ])
        ], style={'padding': '20px', 'backgroundColor': '#f0f0f0', 'borderRadius': '8px', 'marginTop': '30px'})
    ])

layout_analyzer = html.Div(children=[
    html.H1("Option Volatility Smile Analyzer", style={'textAlign': 'center', 'color': '#333'}),
    
    html.Div([
        html.Label("Scrip Name:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Dropdown(
            id='scrip-input',
            options=[{'label': k, 'value': k} for k in sorted(base_name_meta_lookup.keys())],
            value='NIFTY',
            placeholder='Search symbol...',
            searchable=True,
            clearable=False,
            style={'width': '250px', 'fontSize': '16px'}
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
    dcc.Store(id='store-maxpain-data', storage_type='session'),
    dcc.Store(id='sidebar-state', data='open'),

    # Toggle Button (always visible)
    html.Button("☰", id="sidebar-toggle", n_clicks=0, style={
        'position': 'fixed', 'top': '10px', 'left': '10px', 'zIndex': 1100,
        'fontSize': '20px', 'padding': '5px 10px', 'cursor': 'pointer',
        'backgroundColor': '#007bff', 'color': 'white', 'border': 'none',
        'borderRadius': '5px'
    }),

    # Sidebar
    html.Div([
        html.H4("Kite Analytics", style={'marginTop': '40px', 'fontSize': '16px', 'fontWeight': 'bold'}),
        html.Hr(),
        dcc.Link('Home', href='/', style={'display': 'block', 'padding': '8px', 'textDecoration': 'none', 'color': '#333', 'fontSize': '14px'}),
        dcc.Link('Vol Analyzer', href='/analyzer', style={'display': 'block', 'padding': '8px', 'textDecoration': 'none', 'color': '#333', 'fontSize': '14px'}),
        dcc.Link('PCR Skew', href='/pcr', style={'display': 'block', 'padding': '8px', 'textDecoration': 'none', 'color': '#333', 'fontSize': '14px'}),
        dcc.Link('IV Ratio', href='/iv-ratio', style={'display': 'block', 'padding': '8px', 'textDecoration': 'none', 'color': '#333', 'fontSize': '14px'}),
        dcc.Link('Max Pain', href='/max-pain', style={'display': 'block', 'padding': '8px', 'textDecoration': 'none', 'color': '#333', 'fontSize': '14px'}),
        dcc.Link('Calendar IV', href='/calendar', style={'display': 'block', 'padding': '8px', 'textDecoration': 'none', 'color': '#fd7e14', 'fontSize': '14px', 'fontWeight': 'bold'}),
    ], id='sidebar', style=SIDEBAR_STYLE),
    
    # Content
    html.Div(id="page-content", style=CONTENT_STYLE)
])

# --- Clientside callback for sidebar toggle ---
app.clientside_callback(
    """
    function(n_clicks, current_state) {
        if (!n_clicks) return [current_state, {}, {}];
        var new_state = current_state === 'open' ? 'closed' : 'open';
        var sidebar_style, content_style;
        if (new_state === 'closed') {
            sidebar_style = {'position':'fixed','top':0,'left':0,'bottom':0,'width':'0','padding':'0','overflow':'hidden','backgroundColor':'#f8f9fa','transition':'all 0.3s','zIndex':1000};
            content_style = {'marginLeft':'1rem','marginRight':'0.5rem','padding':'1rem 0.5rem','transition':'all 0.3s'};
        } else {
            sidebar_style = {'position':'fixed','top':0,'left':0,'bottom':0,'width':'11rem','padding':'1rem 0.5rem','backgroundColor':'#f8f9fa','transition':'all 0.3s','overflowY':'auto','zIndex':1000};
            content_style = {'marginLeft':'12rem','marginRight':'0.5rem','padding':'1rem 0.5rem','transition':'all 0.3s'};
        }
        return [new_state, sidebar_style, content_style];
    }
    """,
    [Output('sidebar-state', 'data'),
     Output('sidebar', 'style'),
     Output('page-content', 'style')],
    [Input('sidebar-toggle', 'n_clicks')],
    [State('sidebar-state', 'data')]
)


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
    elif pathname == "/max-pain":
        return layout_max_pain_view()
    elif pathname == "/calendar":
        return layout_calendar_view()
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
            marker=dict(color='rgba(40, 167, 69, 0.45)'), # Green with 0.45 opacity
            yaxis='y2'
        ))
        
        # PE OI (Red, Transparent)
        fig.add_trace(go.Bar(
            x=df['STRIKE'], y=df['PE-OI'],
            name='PE OI',
            marker=dict(color='rgba(220, 53, 69, 0.45)'), # Red with 0.45 opacity
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
    df_heat = _get_heatmap_data()
    
    if df_heat.empty:
        return go.Figure(), dash.no_update
    # Use only nearest expiry for IV Skew heatmap
    if 'Expiry_Rank' in df_heat.columns:
        df_heat = df_heat[df_heat['Expiry_Rank'] == 1].copy()
        
    # Bin Moneyness - Granularity 0.02
    bin_step = 0.02
    df_heat['Moneyness_Bin'] = (df_heat['Moneyness'] / bin_step).round() * bin_step
    
    # Define Color Scale (Standard Ratio)
    z_mid = 1.0
    colors = 'RdBu_r'
    title = "IV / ATM IV"

    # Helper to pivot multiple metrics aligned
    def create_pivot(df_sub):
        # Pivot each metric
        p_ratio = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='Ratio', aggfunc='mean').sort_index(ascending=False)
        p_oi = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='OI', aggfunc='mean').sort_index(ascending=False)
        p_bid = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='Bid', aggfunc='mean').sort_index(ascending=False)
        p_ask = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='Ask', aggfunc='mean').sort_index(ascending=False)
        p_strike = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='Strike', aggfunc='mean').sort_index(ascending=False)
        p_iv = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='IV', aggfunc='mean').sort_index(ascending=False)
        
        # Filter columns to 0.8-1.2
        cols = [c for c in p_ratio.columns if 0.8 <= c <= 1.2]
        
        return p_ratio[cols], p_oi[cols], p_bid[cols], p_ask[cols], p_strike[cols], p_iv[cols]

    # --- CE HEATMAP ---
    df_ce = df_heat[df_heat['Type'] == 'CE']
    ce_ratio, ce_oi, ce_bid, ce_ask, ce_strike, ce_iv = create_pivot(df_ce)
    
    # Custom Data for CE: Stack along last axis
    # shape: (rows, cols, 5) -> OI, Bid, Ask, Strike, IV
    import numpy as np
    ce_custom = np.dstack((ce_oi.values, ce_bid.values, ce_ask.values, ce_strike.values, ce_iv.values))
    
    # --- PE HEATMAP ---
    df_pe = df_heat[df_heat['Type'] == 'PE']
    pe_ratio, pe_oi, pe_bid, pe_ask, pe_strike, pe_iv = create_pivot(df_pe)
    
    # Custom Data for PE
    pe_custom = np.dstack((pe_oi.values, pe_bid.values, pe_ask.values, pe_strike.values, pe_iv.values))
    
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
        hovertemplate="<b>Symbol: %{y}</b><br>Moneyness: %{x}<br>Strike: %{customdata[3]:,.0f}<br>IV: %{customdata[4]:.2f}%<br>Ratio: %{z:.2f}<br>OI: %{customdata[0]:,}<br>Bid: %{customdata[1]:.2f}<br>Ask: %{customdata[2]:.2f}<extra>CE</extra>",
        **hm_args
    ), row=1, col=1)
    
    # Add PE Trace
    fig.add_trace(go.Heatmap(
        z=pe_ratio.values,
        x=pe_ratio.columns,
        y=pe_ratio.index,
        customdata=pe_custom,
        colorbar_y=0.2,
        hovertemplate="<b>Symbol: %{y}</b><br>Moneyness: %{x}<br>Strike: %{customdata[3]:,.0f}<br>IV: %{customdata[4]:.2f}%<br>Ratio: %{z:.2f}<br>OI: %{customdata[0]:,}<br>Bid: %{customdata[1]:.2f}<br>Ask: %{customdata[2]:.2f}<extra>PE</extra>",
        **hm_args
    ), row=2, col=1)
    
    # Layout
    fig.update_layout(
        title=f"Market IV Skew Heatmap (All {len(ce_ratio)} Symbols — NFO + MCX + BFO)",
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
    df_heat = _get_heatmap_data()
    
    if df_heat.empty:
        return go.Figure(), dash.no_update
    if 'Expiry_Rank' in df_heat.columns:
        df_heat = df_heat[df_heat['Expiry_Rank'] == 1].copy()
    
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
    df_heat = _get_heatmap_data()
    
    if df_heat.empty:
        return go.Figure(), dash.no_update
    if 'Expiry_Rank' in df_heat.columns:
        df_heat = df_heat[df_heat['Expiry_Rank'] == 1].copy()
    
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

# --- MAX PAIN CALLBACKS ---
@app.callback(
    [Output('maxpain-table-container', 'children'),
     Output('graph-maxpain-heatmap', 'figure'),
     Output('store-maxpain-data', 'data')],
    [Input('btn-maxpain', 'n_clicks')],
    [State('store-maxpain-data', 'data')]
)
def update_maxpain(n_clicks, stored_data):
    if not n_clicks:
        if stored_data:
            # Restore from session store
            summary = pd.DataFrame(stored_data['summary'])
            pain_profiles = {k: pd.DataFrame(v) for k, v in stored_data['profiles'].items()}
        else:
            return html.Div(), go.Figure(), dash.no_update
    else:
        # Get data from disk cache
        df_heat = _get_heatmap_data()
        if df_heat.empty:
            return html.Div("No data available."), go.Figure(), dash.no_update
        if 'Expiry_Rank' in df_heat.columns:
            df_heat = df_heat[df_heat['Expiry_Rank'] == 1].copy()
        
        summary, pain_profiles = compute_max_pain(df_heat)
    
    if summary.empty:
        return html.Div("No Max Pain data computed."), go.Figure(), dash.no_update
    
    # --- Section 1: Summary DataTable ---
    table_df = summary[['Symbol', 'Spot', 'Max Pain', 'Distance%', 'Bias']].copy()
    
    table = dash_table.DataTable(
        id='maxpain-table',
        columns=[{'name': c, 'id': c} for c in table_df.columns],
        data=table_df.to_dict('records'),
        sort_action='native',
        filter_action='native',
        page_size=20,
        row_selectable='single',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '14px'},
        style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'filter_query': '{Bias} contains "Bull"'}, 'backgroundColor': '#d4edda', 'color': '#155724'},
            {'if': {'filter_query': '{Bias} contains "Bear"'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'},
        ]
    )
    
    # --- Section 2: Top 5 Pain Heatmap ---
    # Build matrix: rows = symbols, cols = rank (1st-5th highest pain strike)
    symbols = summary['Symbol'].tolist()
    
    # Z values (total pain), custom data, and x-labels
    z_matrix = []
    custom_matrix = []
    x_labels = ['#1 Highest', '#2', '#3', '#4', '#5']
    
    for _, row in summary.iterrows():
        sym = row['Symbol']
        top5_strikes = row['Top5_Strikes']
        top5_pain = row['Top5_Pain']
        profile = pain_profiles.get(sym, pd.DataFrame())
        
        z_row = []
        custom_row = []
        for j in range(5):
            if j < len(top5_strikes):
                strike = top5_strikes[j]
                pain = top5_pain[j]
                # Get CE/PE breakdown from profile
                prow = profile[profile['Strike'] == strike]
                ce_pain = float(prow['CE_Pain'].iloc[0]) if not prow.empty else 0
                pe_pain = float(prow['PE_Pain'].iloc[0]) if not prow.empty else 0
                ce_oi = float(prow['CE_OI'].iloc[0]) if not prow.empty else 0
                pe_oi = float(prow['PE_OI'].iloc[0]) if not prow.empty else 0
                z_row.append(pain)
                custom_row.append([strike, pain, ce_pain, pe_pain, ce_oi, pe_oi])
            else:
                z_row.append(0)
                custom_row.append([0, 0, 0, 0, 0, 0])
        z_matrix.append(z_row)
        custom_matrix.append(custom_row)
    
    # Normalize per row (0-1) so each symbol uses full color range
    z_normalized = []
    for z_row in z_matrix:
        row_max = max(z_row) if max(z_row) > 0 else 1
        z_normalized.append([v / row_max for v in z_row])
    
    fig_heatmap = go.Figure(go.Heatmap(
        z=z_normalized,
        x=x_labels,
        y=symbols,
        customdata=custom_matrix,
        colorscale='YlOrRd',
        zmin=0, zmax=1,
        colorbar=dict(title='Pain (normalized)', tickvals=[0, 0.5, 1], ticktext=['Low', 'Mid', 'High']),
        hovertemplate="<b>%{y}</b><br>Rank: %{x}<br>Strike: %{customdata[0]}<br>Total Pain: %{customdata[1]:,.0f}<br>CE Pain: %{customdata[2]:,.0f}<br>PE Pain: %{customdata[3]:,.0f}<br>CE OI: %{customdata[4]:,}<br>PE OI: %{customdata[5]:,}<extra></extra>"
    ))
    
    fig_heatmap.update_layout(
        title=f"Top 5 Highest-Pain Strikes ({len(symbols)} Symbols) — Color normalized per symbol",
        height=max(800, len(symbols) * 25),
        yaxis={'dtick': 1}
    )
    
    # Prepare store data (serialize profiles for session storage)
    store_data = {
        'summary': summary.to_dict('records'),
        'profiles': {k: v.to_dict('records') for k, v in pain_profiles.items()}
    }
    
    return table, fig_heatmap, store_data


@app.callback(
    [Output('graph-maxpain-drilldown', 'figure'),
     Output('drilldown-title', 'children')],
    [Input('maxpain-table', 'selected_rows')],
    [State('maxpain-table', 'data'),
     State('store-maxpain-data', 'data')]
)
def update_maxpain_drilldown(selected_rows, table_data, stored_data):
    if not selected_rows or not table_data or not stored_data:
        return go.Figure(), "Click a row in the table to see the full pain profile"
    
    row = table_data[selected_rows[0]]
    sym = row['Symbol']
    spot = row['Spot']
    max_pain = row['Max Pain']
    
    # Get pain profile from stored data
    profiles = stored_data.get('profiles', {})
    if sym not in profiles:
        return go.Figure(), f"No profile data for {sym}"
    
    profile = pd.DataFrame(profiles[sym])
    
    # Stacked bar chart: CE Pain + PE Pain at each strike
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=profile['Strike'], y=profile['CE_Pain'],
        name='CE Pain (Resistance)',
        marker_color='#dc3545',
        hovertemplate="Strike: %{x}<br>CE Pain: %{y:,.0f}<extra>CE</extra>"
    ))
    
    fig.add_trace(go.Bar(
        x=profile['Strike'], y=profile['PE_Pain'],
        name='PE Pain (Support)',
        marker_color='#28a745',
        hovertemplate="Strike: %{x}<br>PE Pain: %{y:,.0f}<extra>PE</extra>"
    ))
    
    # Mark Max Pain strike
    fig.add_vline(
        x=max_pain, line_width=3, line_dash="dash", line_color="blue",
        annotation_text=f"Max Pain: {max_pain}", annotation_position="top right"
    )
    
    # Mark Spot price
    fig.add_vline(
        x=spot, line_width=2, line_dash="dot", line_color="black",
        annotation_text=f"Spot: {spot}", annotation_position="top left"
    )
    
    fig.update_layout(
        barmode='stack',
        title=f"Pain Profile: {sym}  |  Spot: {spot}  |  Max Pain: {max_pain}  |  Distance: {row['Distance%']}%",
        xaxis_title="Strike Price",
        yaxis_title="Total Pain (₹ × OI)",
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig, f"Pain Profile for {sym}"

# --- CALENDAR SPREAD CALLBACKS ---
@app.callback(
    [Output('graph-calendar-atm', 'figure'),
     Output('calendar-table-container', 'children'),
     Output('store-calendar-data', 'data')],
    [Input('btn-calendar', 'n_clicks')],
    [State('store-calendar-data', 'data')]
)
def update_calendar_spread(n_clicks, stored_data):
    if not n_clicks:
        if stored_data:
            # Restore if available (though complex to restore table fully without regenerating)
            # For simplicity, just return empty on load or re-generate if needed.
            # But stored_data is dict of records.
            return go.Figure(), html.Div(), dash.no_update
        return go.Figure(), html.Div(), dash.no_update
    
    # 1. Get Data (Full dataset with all ranks)
    df_heat = _get_heatmap_data()
    
    if df_heat.empty:
        return go.Figure(), html.Div("No data available."), dash.no_update
    
    # Ensure necessary columns
    needed_cols = ['Symbol', 'Expiry_Rank', 'ATM_IV', 'Expiry']
    if not all(c in df_heat.columns for c in needed_cols):
        return go.Figure(), html.Div("Data missing required columns (Expiry_Rank, ATM_IV). Try clearing cache."), dash.no_update
        
    # 2. Process Data: Find symbols with BOTH Rank 1 and Rank 2
    # Group by Symbol and Expiry_Rank to get ATM_IV (should be same for all rows of that rank)
    df_atm = df_heat.groupby(['Symbol', 'Expiry_Rank']).agg({
        'ATM_IV': 'first',
        'Expiry': 'first',
        'Spot': 'first'
    }).reset_index()
    
    # Pivot to get Rank 1 and Rank 2 side-by-side
    df_pivot = df_atm.pivot(index='Symbol', columns='Expiry_Rank', values=['ATM_IV', 'Expiry', 'Spot'])
    
    # Flatten columns
    df_pivot.columns = [f'{c[0]}_{c[1]}' for c in df_pivot.columns]
    
    # Filter for symbols that have both Rank 1 and Rank 2
    if 'ATM_IV_1' not in df_pivot.columns or 'ATM_IV_2' not in df_pivot.columns:
         return go.Figure(), html.Div("Insufficient multi-expiry data found."), dash.no_update
         
    df_spread = df_pivot.dropna(subset=['ATM_IV_1', 'ATM_IV_2']).copy()
    
    if df_spread.empty:
        return go.Figure(), html.Div("No symbols found with both Near and Far expiry data."), dash.no_update
        
    # Calculate Spread: Near IV - Far IV
    # Positive = Near is higher (IV Inversion / Backwardation? / Event) -> Red
    # Negative = Far is higher (Normal / Contango) -> Green
    df_spread['Spread'] = df_spread['ATM_IV_1'] - df_spread['ATM_IV_2']
    df_spread['Abs_Spread'] = df_spread['Spread'].abs()
    
    # Sort by magnitude of spread
    df_spread = df_spread.sort_values('Abs_Spread', ascending=False).reset_index()
    
    # Prepare Table
    table_df = df_spread[['Symbol', 'ATM_IV_1', 'ATM_IV_2', 'Spread', 'Expiry_1', 'Expiry_2']].copy()
    table_df.columns = ['Symbol', 'Near IV%', 'Far IV%', 'Spread', 'Near Exp', 'Far Exp']
    table_df['Near IV%'] = table_df['Near IV%'].round(2)
    table_df['Far IV%'] = table_df['Far IV%'].round(2)
    table_df['Spread'] = table_df['Spread'].round(2)
    
    table = dash_table.DataTable(
        id='calendar-table',
        columns=[{'name': c, 'id': c} for c in table_df.columns],
        data=table_df.to_dict('records'),
        sort_action='native',
        filter_action='native',
        page_size=15,
        row_selectable='single',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '8px'},
        style_header={'backgroundColor': '#fd7e14', 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'filter_query': '{Spread} > 0', 'column_id': 'Spread'}, 'color': '#dc3545', 'fontWeight': 'bold'}, # Red for Inversion
            {'if': {'filter_query': '{Spread} < 0', 'column_id': 'Spread'}, 'color': '#28a745', 'fontWeight': 'bold'}, # Green for Normal
        ]
    )
    
    # Prepare Bar Chart (Top 30 by spread magnitude)
    top_n = df_spread.head(30)
    colors = ['#dc3545' if s > 0 else '#28a745' for s in top_n['Spread']] # Red if > 0, Green if < 0
    
    fig = go.Figure(go.Bar(
        x=top_n['Spread'],
        y=top_n['Symbol'],
        orientation='h',
        marker_color=colors,
        text=top_n['Spread'].round(2),
        textposition='auto',
        hovertemplate="<b>%{y}</b><br>Spread: %{x:.2f}%<br>Near IV: %{customdata[0]:.2f}%<br>Far IV: %{customdata[1]:.2f}%<extra></extra>",
        customdata=top_n[['ATM_IV_1', 'ATM_IV_2']]
    ))
    
    fig.update_layout(
        title=f"Calendar Spread (Near IV - Far IV) — Top {len(top_n)} Divergences",
        xaxis_title="Spread (IV%)",
        yaxis={'autorange': 'reversed'},
        height=max(600, len(top_n) * 20),
        template="plotly_white"
    )
    
    fig.add_vline(x=0, line_width=1, line_color="black")
    
    # Store full data for drilldown (convert to dict of records)
    # We only need rows for symbols in our spread list to save space
    relevant_symbols = df_spread['Symbol'].tolist()
    df_store = df_heat[df_heat['Symbol'].isin(relevant_symbols)].copy()
    
    # Store as simple list of records
    store_data = df_store.to_dict('records')
    
    return fig, table, store_data


@app.callback(
    [Output('graph-calendar-drilldown', 'figure'),
     Output('calendar-drilldown-title', 'children')],
    [Input('calendar-table', 'selected_rows')],
    [State('calendar-table', 'data'),
     State('store-calendar-data', 'data')]
)
def update_calendar_drilldown(selected_rows, table_data, stored_data):
    if not selected_rows or not table_data or not stored_data:
        return go.Figure(), "Click a row in the table to see strike-level details"
    
    row = table_data[selected_rows[0]]
    sym = row['Symbol']
    
    # Filter stored data
    df = pd.DataFrame(stored_data)
    df_sym = df[df['Symbol'] == sym].copy()
    
    if df_sym.empty:
        return go.Figure(), f"No data found for {sym}"
    
    # Separate rankings
    df1 = df_sym[df_sym['Expiry_Rank'] == 1].sort_values('Strike')
    df2 = df_sym[df_sym['Expiry_Rank'] == 2].sort_values('Strike')
    
    # Identify expiries
    exp1 = df1['Expiry'].iloc[0] if not df1.empty else "Near"
    exp2 = df2['Expiry'].iloc[0] if not df2.empty else "Far"
    
    # Figure: Grouped Bar
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df1['Strike'], y=df1['IV'],
        name=f"Near: {exp1}",
        marker_color='#007bff'
    ))
    
    fig.add_trace(go.Bar(
        x=df2['Strike'], y=df2['IV'],
        name=f"Far: {exp2}",
        marker_color='#fd7e14' # Orange
    ))
    
    # Calculate difference at each common strike
    common_strikes = set(df1['Strike']).intersection(set(df2['Strike']))
    # Could add a line for difference? Maybe too messy.
    # Let's keep it simple side-by-side bars.
    
    fig.update_layout(
        title=f"Calendar IV Structure: {sym} ({exp1} vs {exp2})",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility (%)",
        barmode='group',
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig, f"Strike-wise IV Comparison: {sym}"
# ==============================================================================
# 5. SERVER RUN
# ==============================================================================
if __name__ == '__main__':
    # debug=False is safer when using global variables in complex scripts
    app.run(debug=True, use_reloader=False)