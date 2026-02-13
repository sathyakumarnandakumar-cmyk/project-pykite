# ==============================================================================
# 1. IMPORTS & SETUP
# ==============================================================================
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
import tempfile
import warnings
import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sqlite3
import threading

# Volatility Imports
from py_vollib.black_scholes.implied_volatility import implied_volatility as bsm_iv

# --- CONFIGURATION ---
IV_LOG_FILE = 'iv_log_data.csv'  # File to store the 1-minute interval data
IV_DB_FILE = 'iv_storage.db'      # SQLite DB for IV Pro Trader

# --- DATABASE ENGINE (SQLite) ---
def init_db():
    try:
        conn = sqlite3.connect(IV_DB_FILE)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS iv_history (
                timestamp TEXT,
                symbol TEXT,
                expiry TEXT,
                strike REAL,
                ce_iv REAL,
                pe_iv REAL,
                spot REAL
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sym_strike ON iv_history (symbol, strike);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON iv_history (timestamp);')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[X] DB Init Error: {e}")

def cleanup_expired_data():
    try:
        conn = sqlite3.connect(IV_DB_FILE)
        cursor = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("DELETE FROM iv_history WHERE expiry < ?", (today,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[X] DB Cleanup Error: {e}")

# --- MARKET HOURS HELPER ---
def is_market_open(exchange='NFO'):
    """Check if market is open for the given exchange.
    NFO/BFO/NSE: 9:15 - 15:30 IST (weekdays)
    MCX:         9:00 - 23:30 IST (weekdays)
    Returns (is_open: bool, reason: str)
    """
    now = datetime.now()
    weekday = now.weekday()  # 0=Mon .. 6=Sun
    if weekday >= 5:
        return False, "Weekend — markets closed"
    t = now.hour * 60 + now.minute  # minutes since midnight
    if exchange == 'MCX':
        if t < 9 * 60 or t > 23 * 60 + 30:
            return False, f"MCX closed (09:00-23:30). Current: {now.strftime('%H:%M')}"
    else:  # NFO, BFO, NSE
        if t < 9 * 60 + 15 or t > 15 * 60 + 30:
            return False, f"{exchange} closed (09:15-15:30). Current: {now.strftime('%H:%M')}"
    return True, "Market open"

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
    
    # 4. Filter for Options (NFO & MCX only) - Needed for IV Pro Recorder
    df_options = df_all[
        (df_all['instrument_type'].isin(['CE', 'PE'])) & 
        (df_all['exchange'].isin(['NFO', 'MCX']))
    ].copy()
    df_options['expiry'] = pd.to_datetime(df_options['expiry'])

    # ==============================================================================
    # 3. BACKGROUND DATA COLLECTOR (The "Smart Filter" Logger from livesmart.py)
    # ==============================================================================
    
    def get_active_contracts():
        """
        Returns a DataFrame of ONLY the 'Current Expiry' contracts for all symbols.
        This reduces the scanning universe from 50k to ~10k relevant items.
        """
        relevant_tokens = []
        grouped = df_options.groupby('name')
        
        for name, group in grouped:
            expiries = sorted(group['expiry'].unique())
            if not expiries: continue
            
            # Select Nearest Expiry Only
            curr_expiry = expiries[0]
            mask = group['expiry'] == curr_expiry
            relevant_tokens.append(group[mask])
            
        if not relevant_tokens: return pd.DataFrame()
        return pd.concat(relevant_tokens)

    def data_recorder_loop():
        """
        Smart Recorder: 
        1. Fetches ALL active current-expiry contracts.
        2. Filters Top 5 OI (CE/PE) + ATM per symbol.
        3. Calculates IV and Stores.
        """
        print(" [LOGGER] Smart Data Recorder Started (Tracking Top 5 OI + ATM)...")
        init_db()
        cleanup_expired_data()
        
        # Pre-calculate active tokens to scan
        df_active = get_active_contracts()
        print(f" [LOGGER] Tracking {len(df_active)} active contracts across {df_active['name'].nunique()} symbols.")
        
        # Spot mapping
        unique_names = df_active['name'].unique()
        spot_map = {} 
        for name in unique_names:
             # Use base_name_meta_lookup for exchange info
             ex = base_name_meta_lookup.get(name, {}).get('exchange', 'NSE')
             if name == 'NIFTY': s = "NSE:NIFTY 50"
             elif name == 'BANKNIFTY': s = "NSE:NIFTY BANK"
             elif name == 'FINNIFTY': s = "NSE:NIFTY FIN SERVICE"
             elif ex == 'MCX':
                 # Dynamically find nearest futures contract for MCX
                 mcx_fut_exps = base_name_meta_lookup.get(name, {}).get('futures_expiries', [])
                 if mcx_fut_exps:
                     # Find symbol for this expiry
                     fut_row = df_all[(df_all['name'] == name) & (df_all['instrument_type'] == 'FUT') & (df_all['expiry'] == mcx_fut_exps[0])]
                     if not fut_row.empty:
                         s = f"MCX:{fut_row.iloc[0]['tradingsymbol']}"
                     else:
                         s = f"MCX:{name}"
                 else:
                     s = f"MCX:{name}"
             else: s = f"NSE:{name}" # Stocks
             spot_map[name] = s

        while True:
            now = datetime.now()
            
            # --- MARKET HOURS GATE ---
            # NFO/BFO: 9:15 - 15:30 IST | MCX: 9:00 - 23:30 IST
            
            # Broadest window check first (MCX start to end)
            weekday = now.weekday()
            current_hour = now.hour
            current_minute = now.minute
            current_time_mins = current_hour * 60 + current_minute
            
            market_open = 9 * 60        # 09:00
            market_close = 23 * 60 + 30 # 23:30
            
            if weekday >= 5 or current_time_mins < market_open or current_time_mins > market_close:
                # Still check if we are in expected trading days to avoid spam logs
                if weekday >= 5:
                    # Weekend - print less frequent or just sleep
                    # print(f" [LOGGER] Weekend detected. Sleeping 60s...")
                    pass
                else:
                    print(f" [LOGGER] Outside market hours ({now.strftime('%H:%M')}). Sleeping 60s...")
                time.sleep(60)
                continue
            
            start_time = time.time()
            now_str = now.strftime('%Y-%m-%d %H:%M:%S')
            
            # --- A. PREPARE BATCH FETCH ---
            # 1. Option Tokens
            all_instruments = [f"{row['exchange']}:{row['tradingsymbol']}" for _, row in df_active.iterrows()]
            
            # 2. Spot Tokens
            spot_symbols = list(spot_map.values())
            all_tokens_to_fetch = list(set(all_instruments + spot_symbols))
            
            # --- B. FETCH DATA (BATCHED) ---
            market_data = {}
            chunk_size = 500
            
            for i in range(0, len(all_tokens_to_fetch), chunk_size):
                try:
                    batch = all_tokens_to_fetch[i:i+chunk_size]
                    quotes = kite.quote(batch)
                    market_data.update(quotes)
                except Exception as e:
                    pass

            # --- C. PROCESS LOOP ---
            
            # NFO/BFO/NSE specific gates
            nfo_closed = current_time_mins > (15 * 60 + 30)
            nfo_not_open = current_time_mins < (9 * 60 + 15)
            
            records_to_insert = []
            for symbol in unique_names:
                # Skip NFO/BFO/NSE symbols outside their market hours
                sym_exchange = base_name_meta_lookup.get(symbol, {}).get('exchange', '')
                if sym_exchange in ('NFO', 'BFO', 'NSE') and (nfo_closed or nfo_not_open):
                    continue
                
                # 1. Get Spot Price
                spot_key = spot_map.get(symbol)
                if spot_key not in market_data: continue
                spot_price = market_data[spot_key]['last_price']
                if spot_price == 0: continue
                
                # 2. Get Candidates (Current Expiry Only)
                sym_df = df_active[df_active['name'] == symbol]
                if sym_df.empty: continue
                
                expiry = sym_df.iloc[0]['expiry']
                exchange = sym_df.iloc[0]['exchange']
                
                # Enrich with live data (OI & Price)
                candidates = []
                for _, row in sym_df.iterrows():
                    key = f"{exchange}:{row['tradingsymbol']}"
                    if key in market_data:
                        q = market_data[key]
                        candidates.append({
                            'strike': row['strike'],
                            'type': row['instrument_type'],
                            'oi': q['oi'],
                            'price': q['last_price'],
                            'depth': q['depth'],
                            'last_price': q['last_price']
                        })
                
                if not candidates: continue
                cand_df = pd.DataFrame(candidates)
                
                # 3. FILTER LOGIC (Top 5 OI + ATM)
                # Find ATM
                atm_strike = cand_df.iloc[(cand_df['strike'] - spot_price).abs().argsort()[:1]]['strike'].values[0]
                
                # Top 5 by OI
                top_ce = cand_df[cand_df['type'] == 'CE'].nlargest(5, 'oi')
                top_pe = cand_df[cand_df['type'] == 'PE'].nlargest(5, 'oi')
                
                # Union of strikes (Indices)
                target_indices = set(top_ce.index).union(set(top_pe.index))
                target_indices = target_indices.union(set(cand_df[cand_df['strike'] == atm_strike].index))
                
                final_selection = cand_df.loc[list(target_indices)]
                
                # 4. CALCULATE IV & AGGREGATE
                strike_map = {} # strike -> {ce: 0, pe: 0}
                
                t = (expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
                if t <= 0: t = 0.0001
                
                for _, row in final_selection.iterrows():
                    iv = 0
                    strike = float(row['strike'])
                    flag = 'c' if row['type'] == 'CE' else 'p'
                    
                    # Get Mid Price
                    bid = row['depth']['buy'][0]['price'] if row['depth']['buy'] else 0
                    ask = row['depth']['sell'][0]['price'] if row['depth']['sell'] else 0
                    price = (bid + ask) / 2 if (bid > 0 and ask > 0) else row['last_price']
                    
                    # IV Calc
                    if price > 0.05:
                        intrinsic = max(0, spot_price - strike) if flag == 'c' else max(0, strike - spot_price)
                        if price > intrinsic:
                            try:
                                iv = bsm_iv(price, spot_price, strike, t, 0.10, flag) * 100
                            except: iv = 0
                    
                    if strike not in strike_map: strike_map[strike] = {'ce': 0, 'pe': 0}
                    if row['type'] == 'CE': strike_map[strike]['ce'] = iv
                    else: strike_map[strike]['pe'] = iv
    
                # 5. PREPARE DB ROWS
                for s, vals in strike_map.items():
                    if vals['ce'] > 0 or vals['pe'] > 0:
                        records_to_insert.append((
                            now_str, symbol, str(expiry.date()), s, vals['ce'], vals['pe'], spot_price
                        ))
            
            # --- D. BULK INSERT ---
            if records_to_insert:
                try:
                    conn = sqlite3.connect(IV_DB_FILE)
                    conn.executemany("INSERT INTO iv_history VALUES (?,?,?,?,?,?,?)", records_to_insert)
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f" [ERROR] DB Insert failed: {e}")
    
            # Sleep remaining time to make it ~1 minute interval
            elapsed = time.time() - start_time
            sleep_time = max(0, 60 - elapsed)
            time.sleep(sleep_time)
            
    # Start Background Thread
    threading.Thread(target=data_recorder_loop, daemon=True).start()
    
    print(f"[OK] Initialization Complete. Loaded {len(df_all)} instruments.")

except Exception as e:
    print(f"[X] CRITICAL ERROR during data loading: {e}")
    sys.exit(1)

# ==============================================================================
# 3. CORE FUNCTIONS
# ==============================================================================

def build_option_chain(symbol, expiry=None, enrich=False):
    if symbol not in base_name_meta_lookup:
        return {}, pd.DataFrame()
    
    entry = base_name_meta_lookup[symbol]
    exchange = entry['exchange']
    options_expiries = entry['options_expiries']
    
    if not options_expiries:
        return {}, pd.DataFrame()
    
    if expiry is None:
        expiry = options_expiries[0] # Nearest
    else:
        expiry = pd.to_datetime(expiry)
    
    if exchange == 'MCX': source_df = df_mcx
    elif exchange == 'BFO': source_df = df_bfo
    else: source_df = df_all
    
    mask = (source_df['name'] == symbol) & (source_df['expiry'] == expiry) & (source_df['instrument_type'].isin(['CE', 'PE']))
    chain = source_df[mask].copy()
    
    if chain.empty: return {}, pd.DataFrame()
    
    cols = ['tradingsymbol', 'strike', 'instrument_type', 'expiry', 'lot_size', 'instrument_token', 'tick_size', 'segment']
    chain = chain[[c for c in cols if c in chain.columns]].sort_values('strike').reset_index(drop=True)
    
    metadata = {
        'basename': symbol, 'exchange': exchange, 'expiry': expiry,
        'lot_size': int(chain['lot_size'].iloc[0]) if 'lot_size' in chain.columns else None
    }
    
    if enrich:
        metadata, chain = enrich_with_market_data(kite, chain, metadata)
        
    return metadata, chain

def enrich_with_market_data(kite_obj, option_chain_df, metadata, chunk_size=250):
    if option_chain_df.empty: return metadata, option_chain_df
    exchange = metadata.get('exchange', 'NFO')
    enriched_data = []
    tradingsymbols = option_chain_df['tradingsymbol'].tolist()
    symbols_to_quote = [f"{exchange}:{sym}" for sym in tradingsymbols]
    all_quotes = {}
    
    for i in range(0, len(symbols_to_quote), chunk_size):
        try:
            quotes = kite_obj.quote(symbols_to_quote[i:i+chunk_size])
            all_quotes.update(quotes)
        except Exception as e: print(f"Error fetching quotes: {e}")
            
    for idx, row in option_chain_df.iterrows():
        key = f"{exchange}:{row['tradingsymbol']}"
        quote = all_quotes.get(key, {})
        depth = quote.get('depth', {})
        buy_depth = depth.get('buy', [])
        sell_depth = depth.get('sell', [])
        bid = buy_depth[0]['price'] if buy_depth else 0
        ask = sell_depth[0]['price'] if sell_depth else 0
        ltp = quote.get('last_price', 0)
        
        if bid > 0 and ask > 0: mid_price = (bid + ask) / 2
        elif bid > 0: mid_price = bid
        elif ask > 0: mid_price = ask
        else: mid_price = ltp
            
        item = row.to_dict()
        item.update({'ltp': ltp, 'mid_price': mid_price, 'oi': quote.get('oi', 0), 'volume': quote.get('volume', 0), 'bid': bid, 'ask': ask})
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
            if basename in base_name_meta_lookup:
               fut_expiries = base_name_meta_lookup[basename].get('futures_expiries', [])
               if fut_expiries:
                   nearest_fut_expiry = fut_expiries[0]
                   f_mask = (df_all['name'] == basename) & (df_all['instrument_type'] == 'FUT') & (df_all['expiry'] == nearest_fut_expiry)
                   f_row = df_all[f_mask]
                   if not f_row.empty:
                       fut_sym = f"MCX:{f_row.iloc[0]['tradingsymbol']}"
                       ltp_data = kite.ltp(fut_sym)
                       spot_price = ltp_data[fut_sym]['last_price']
                   else:
                       spot_price = option_chain['mid_price'].mean()
               else:
                   spot_price = option_chain['mid_price'].mean()
            else:
                spot_price = option_chain['mid_price'].mean()

        elif exchange == 'BFO':
            if basename == 'SENSEX': spot_sym = "BSE:SENSEX"
            elif basename == 'BANKEX': spot_sym = "BSE:BANKEX"
            else: spot_sym = f"BSE:{basename}"
            try:
                ltp_data = kite.ltp(spot_sym)
                spot_price = ltp_data[spot_sym]['last_price']
            except:
                 spot_price = option_chain['mid_price'].mean()

        else:
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
        
    if spot_price == 0: return option_chain, 0, 0
    
    # Time to expiry
    expiry = option_chain['expiry'].iloc[0]
    expiry_dt = expiry + pd.Timedelta(hours=15, minutes=30)
    now = pd.Timestamp.now()
    t = (expiry_dt - now).total_seconds() / (365.25 * 24 * 3600)
    if t <= 0: t = 1 / 365.25
    
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
                
                if strike <= 0:
                    ivs.append(0); continue
                if bid <= 0 or ask <= 0:
                    ivs.append(0); continue
                if price < 0.05:
                    ivs.append(0); continue
                
                if flag == 'c':
                    intrinsic = max(0, spot_price - strike)
                else:
                    intrinsic = max(0, strike - spot_price)
                
                if price <= intrinsic + 0.01:
                    ivs.append(0); continue
                
                iv = bsm_iv(price, spot_price, strike, t, risk_free_rate, flag)
                iv_pct = iv * 100
                
                if iv_pct < 0 or iv_pct > 500:
                    ivs.append(0); continue
                
                ivs.append(iv_pct)
                
            except ValueError:
                ivs.append(0)
            except Exception as e:
                ivs.append(0)
                
    option_chain['iv'] = ivs
    
    # --- ATM IV CALCULATION ---
    atm_iv = 0
    try:
        if spot_price > 0:
            closest_strike_idx = (option_chain['strike'] - spot_price).abs().idxmin()
            atm_strike = option_chain.loc[closest_strike_idx, 'strike']
            atm_app_rows = option_chain[option_chain['strike'] == atm_strike]
            ce_iv = atm_app_rows[atm_app_rows['instrument_type'] == 'CE']['iv'].max()
            pe_iv = atm_app_rows[atm_app_rows['instrument_type'] == 'PE']['iv'].max()
            if pd.isna(ce_iv): ce_iv = 0
            if pd.isna(pe_iv): pe_iv = 0
            if ce_iv > 0 and pe_iv > 0:
                atm_iv = (ce_iv + pe_iv) / 2
            elif ce_iv > 0:
                atm_iv = ce_iv
            elif pe_iv > 0:
                atm_iv = pe_iv
    except Exception as e:
        pass

    return option_chain, spot_price, atm_iv

def show_chain(chain, name):
    if chain.empty: return pd.DataFrame()
    ce_df = chain[chain['instrument_type'] == 'CE'].copy().rename(columns={'iv': 'CE-IV', 'ltp': 'CE-LTP', 'oi': 'CE-OI', 'bid': 'CE-BID', 'ask': 'CE-ASK'})
    pe_df = chain[chain['instrument_type'] == 'PE'].copy().rename(columns={'iv': 'PE-IV', 'ltp': 'PE-LTP', 'oi': 'PE-OI', 'bid': 'PE-BID', 'ask': 'PE-ASK'})
    merged = pd.merge(ce_df[['strike', 'CE-IV', 'CE-LTP', 'CE-OI', 'CE-BID', 'CE-ASK']], pe_df[['strike', 'PE-IV', 'PE-LTP', 'PE-OI', 'PE-BID', 'PE-ASK']], on='strike', how='outer')
    return merged.sort_values('strike').replace({'CE-IV': {0: np.nan}, 'PE-IV': {0: np.nan}}).rename(columns={'strike': 'STRIKE'})

def get_option_data(scrip_name, expiry=None):
    scrip_name = scrip_name.upper().strip()
    meta, chain = build_option_chain(scrip_name, expiry=expiry, enrich=True)
    if chain.empty: return pd.DataFrame(), 0, 0
    chain, spot_price, atm_iv = calculate_iv_vollib(chain, metadata=meta)
    return show_chain(chain, scrip_name), spot_price, atm_iv

# --- Heatmap & Helper Functions ---
def generate_market_heatmap_data(limit=150):
    heatmap_data = []
    all_symbols = []
    for k, v in base_name_meta_lookup.items():
        exchange = v.get('exchange', '')
        if exchange in ('NFO', 'MCX', 'BFO'):
            all_symbols.append((k, exchange))
    if limit:
        all_symbols = all_symbols[:limit]
    print(f"Generating Heatmap for {len(all_symbols)} symbols (NFO + MCX + BFO), 2 expiries each...")
    total = len(all_symbols)
    for i, (sym, exchange) in enumerate(all_symbols):
        # --- Per-symbol market hours gate ---
        mkt_open, mkt_reason = is_market_open(exchange)
        if not mkt_open:
            print(f"\n[SKIP] {sym} ({exchange}): {mkt_reason}")
            continue
        print(f"[{i+1}/{total}] Processing {sym} ({exchange})...", end='\r')
        time.sleep(0.01)
        try:
            entry = base_name_meta_lookup.get(sym, {})
            expiries = entry.get('options_expiries', [])
            if not expiries:
                continue
            expiries_to_process = expiries[:2]
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
                        'Symbol': sym, 'Exchange': exchange, 'Expiry_Rank': exp_rank,
                        'Expiry': str(exp_date.date()) if hasattr(exp_date, 'date') else str(exp_date),
                        'Strike': strike, 'Spot': spot_price, 'Moneyness': moneyness,
                        'Ratio': ratio, 'IV': iv, 'ATM_IV': atm_iv,
                        'Bid': row['bid'], 'Ask': row['ask'], 'Spread': spread,
                        'OI': oi, 'Type': row['instrument_type']
                    })
        except Exception as e:
            print(f"\n[!] Error processing {sym}: {e}")
            continue
    return pd.DataFrame(heatmap_data)

# --- DISK CACHE for Heatmap Data ---
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
    """Compute Max Pain for each symbol from cached heatmap data."""
    results = []
    pain_profiles = {}
    for sym, grp in df_heat.groupby('Symbol'):
        spot = grp['Spot'].iloc[0]
        strikes = sorted(grp['Strike'].unique())
        ce_data = grp[grp['Type'] == 'CE'][['Strike', 'OI']].groupby('Strike')['OI'].sum()
        pe_data = grp[grp['Type'] == 'PE'][['Strike', 'OI']].groupby('Strike')['OI'].sum()
        pain_at_strike = []
        for K in strikes:
            ce_pain = sum(ce_data.get(s, 0) * max(0, K - s) for s in ce_data.index)
            pe_pain = sum(pe_data.get(s, 0) * max(0, s - K) for s in pe_data.index)
            total = ce_pain + pe_pain
            pain_at_strike.append({
                'Strike': K, 'CE_Pain': ce_pain, 'PE_Pain': pe_pain,
                'Total_Pain': total, 'CE_OI': ce_data.get(K, 0), 'PE_OI': pe_data.get(K, 0)
            })
        pdf = pd.DataFrame(pain_at_strike)
        pain_profiles[sym] = pdf
        if pdf.empty: continue
        mp_row = pdf.loc[pdf['Total_Pain'].idxmin()]
        max_pain_strike = mp_row['Strike']
        distance_pct = round((max_pain_strike - spot) / spot * 100, 2)
        top5 = pdf.nlargest(5, 'Total_Pain')
        results.append({
            'Symbol': sym, 'Spot': round(spot, 2), 'Max Pain': round(max_pain_strike, 2),
            'Distance%': distance_pct,
            'Bias': '🟢 Bull' if distance_pct > 0 else '🔴 Bear',
            'Top5_Strikes': top5['Strike'].tolist(), 'Top5_Pain': top5['Total_Pain'].tolist()
        })
    summary = pd.DataFrame(results).sort_values('Symbol')
    return summary, pain_profiles

# ==============================================================================
# 4. DASH WEB APP CONFIGURATION
# ==============================================================================
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Kite Volatility Pro"

# --- Force inline search in scrip dropdown (Dash 4.0) ---
app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
            /* ====== GLOBAL STYLES ====== */
            * { box-sizing: border-box; }
            body { font-family: 'Inter', 'Segoe UI', sans-serif; background: #0a0a14; margin: 0; padding: 0; color: #e0e0e0; -webkit-font-smoothing: antialiased; }
            
            /* ====== REUSABLE DARK THEME COMPONENTS ====== */
            .dark-card {
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 14px;
                padding: 28px;
                backdrop-filter: blur(12px);
                transition: border-color 0.3s ease, box-shadow 0.3s ease;
            }
            .dark-card:hover {
                border-color: rgba(0,230,118,0.15);
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            }
            .page-title {
                font-size: 26px;
                font-weight: 700;
                letter-spacing: 0.5px;
                background: linear-gradient(135deg, #00e676 0%, #00bcd4 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin: 0 0 6px 0;
            }
            .page-subtitle {
                color: rgba(255,255,255,0.45);
                font-size: 13px;
                letter-spacing: 0.3px;
                margin: 0;
            }
            .btn-glow {
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 600;
                font-family: 'Inter', sans-serif;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.25s ease;
                letter-spacing: 0.3px;
                background: linear-gradient(135deg, #00e676 0%, #00c853 100%);
                color: #0a0a14;
                box-shadow: 0 4px 15px rgba(0,230,118,0.25);
            }
            .btn-glow:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 25px rgba(0,230,118,0.4);
            }
            .btn-glow:active { transform: translateY(0); }
            .btn-glow-blue {
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 600;
                font-family: 'Inter', sans-serif;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.25s ease;
                background: linear-gradient(135deg, #2979ff 0%, #1565c0 100%);
                color: #fff;
                box-shadow: 0 4px 15px rgba(41,121,255,0.25);
            }
            .btn-glow-blue:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 25px rgba(41,121,255,0.4);
            }
            .btn-glow-purple {
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 600;
                font-family: 'Inter', sans-serif;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.25s ease;
                background: linear-gradient(135deg, #b388ff 0%, #7c4dff 100%);
                color: #fff;
                box-shadow: 0 4px 15px rgba(124,77,255,0.25);
            }
            .btn-glow-purple:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 25px rgba(124,77,255,0.4);
            }
            .btn-glow-orange {
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 600;
                font-family: 'Inter', sans-serif;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.25s ease;
                background: linear-gradient(135deg, #ffab40 0%, #ff6d00 100%);
                color: #fff;
                box-shadow: 0 4px 15px rgba(255,109,0,0.25);
            }
            .btn-glow-orange:hover {
                transform: translateY(-1px);
                box-shadow: 0 6px 25px rgba(255,109,0,0.4);
            }
            .btn-hint {
                font-size: 11px;
                color: rgba(255,255,255,0.35);
                font-style: italic;
                margin-left: 12px;
            }
            .info-box {
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 10px;
                padding: 20px 24px;
                margin-top: 28px;
                color: rgba(255,255,255,0.6);
                font-size: 13px;
                line-height: 1.7;
            }
            .info-box h4 {
                color: #e0e0e0;
                margin: 0 0 10px 0;
                font-size: 15px;
                font-weight: 600;
            }
            .info-box li { margin-bottom: 4px; }

            /* ====== HERO BANNER ====== */
            .hero-banner {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                border-radius: 16px;
                padding: 36px 36px 32px;
                border: 1px solid rgba(0,230,118,0.1);
                margin-bottom: 24px;
                position: relative;
                overflow: hidden;
            }
            .hero-banner::before {
                content: '';
                position: absolute;
                top: -50%;
                right: -20%;
                width: 400px;
                height: 400px;
                background: radial-gradient(circle, rgba(0,230,118,0.06) 0%, transparent 70%);
                pointer-events: none;
            }
            .hero-title {
                font-size: 28px;
                font-weight: 300;
                color: #fff;
                margin: 0 0 4px 0;
                letter-spacing: 0.5px;
            }
            .hero-title strong {
                font-weight: 700;
                color: #00e676;
            }
            .hero-sub {
                color: rgba(255,255,255,0.4);
                font-size: 13px;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin: 0;
            }

            /* ====== FEATURE GRID ====== */
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 16px;
                margin-bottom: 24px;
            }
            .feature-card {
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 12px;
                padding: 22px 20px;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: block;
            }
            .feature-card:hover {
                background: rgba(255,255,255,0.06);
                border-color: rgba(0,230,118,0.2);
                transform: translateY(-2px);
                box-shadow: 0 8px 24px rgba(0,0,0,0.3);
            }
            .feature-icon {
                font-size: 28px;
                margin-bottom: 10px;
                display: block;
            }
            .feature-name {
                font-size: 15px;
                font-weight: 600;
                color: #e0e0e0;
                margin-bottom: 4px;
            }
            .feature-desc {
                font-size: 12px;
                color: rgba(255,255,255,0.35);
                line-height: 1.4;
            }

            /* ====== SIDEBAR ====== */
            .sidebar-dark {
                position: fixed;
                top: 0; left: 0; bottom: 0;
                width: 220px;
                background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%);
                border-right: 1px solid rgba(255,255,255,0.06);
                padding: 20px 0;
                z-index: 1000;
                overflow-y: auto;
                transition: all 0.3s ease;
            }
            .sidebar-logo {
                padding: 8px 20px 20px;
                border-bottom: 1px solid rgba(255,255,255,0.06);
                margin-bottom: 8px;
            }
            .sidebar-logo-text {
                font-size: 18px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            .sidebar-logo-sub {
                font-size: 10px;
                color: rgba(255,255,255,0.3);
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-top: 2px;
            }
            .nav-link {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 10px 20px;
                color: rgba(255,255,255,0.55);
                text-decoration: none;
                font-size: 13px;
                font-weight: 500;
                transition: all 0.2s ease;
                border-left: 3px solid transparent;
            }
            .nav-link:hover {
                color: #fff;
                background: rgba(255,255,255,0.04);
                border-left-color: #00e676;
            }
            .nav-icon { font-size: 16px; width: 22px; text-align: center; }

            /* ====== ANALYZER PAGE ====== */
            .analyzer-controls {
                display: flex;
                align-items: center;
                gap: 14px;
                flex-wrap: wrap;
                padding: 20px 24px;
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 12px;
                margin-bottom: 24px;
            }
            .analyzer-controls label {
                font-size: 13px;
                font-weight: 600;
                color: rgba(255,255,255,0.6);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            /* ====== IV Pro Control Styles (KEEP) ====== */
            .ctrl-group { display: flex; flex-direction: column; }
            .ctrl-label { font-size: 12px; font-weight: 600; color: #aaa; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
            
            /* --- Proven Fix from livesmart.py --- */
            #ivpro-dd-symbol, #ivpro-dd-expiry, #ivpro-dd-strike,
            .ctrl-group .dash-dropdown,
            .ctrl-group .Select,
            .ctrl-group [class*="container"],
            .ctrl-group [class*="control"],
            .ctrl-group [class*="singleValue"],
            .ctrl-group [class*="option"],
            .ctrl-group [class*="ValueContainer"],
            .ctrl-group [class*="Input"] input,
            .ctrl-group .dash-dropdown span,
            .ctrl-group .dash-dropdown div {
                color: #111 !important;
            }
            .ctrl-group [class*="placeholder"] {
                color: #999 !important;
            }
            .ctrl-group [class*="indicatorSeparator"] {
                background-color: #ccc !important;
            }
            
            /* Dropdown option hover & focus highlight */
            [class*="option"]:hover,
            [id*="option"]:hover {
                background: #dbeafe !important;
                color: #111 !important;
                cursor: pointer;
            }
            [class*="option"][class*="focused"],
            [class*="option"][class*="Focused"],
            [id*="option"][class*="focused"],
            [id*="option"][class*="Focused"] {
                background: #dbeafe !important;
                color: #111 !important;
            }
            [class*="option"][class*="selected"],
            [class*="option"][class*="Selected"],
            [id*="option"][class*="selected"],
            [id*="option"][class*="Selected"] {
                background: #007bff !important;
                color: #fff !important;
            }
            
            /* Timeframe Buttons */
            .time-btn { background: rgba(255,255,255,0.05); color: #aaa; border: 1px solid rgba(255,255,255,0.1); padding: 6px 12px; border-radius: 4px; cursor: pointer; transition: all 0.2s; font-size: 13px; font-weight: 500; }
            .time-btn:hover { background: rgba(255,255,255,0.1); color: #fff; border-color: rgba(255,255,255,0.2); }
            
            /* Chart Cards */
            .chart-card { background: #111; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }

            /* Dropdown Inline Search Fixes */
            #scrip-input input[role='combobox'],
            #scrip-input input[aria-autocomplete='list'] {
                opacity: 1 !important;
            }
            #scrip-input:focus-within [class*='singleValue'] {
                display: none !important;
            }

            /* ====== SCROLLBAR ====== */
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: #0a0a14; }
            ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
            .modebar { opacity: 0.3 !important; }
            .modebar:hover { opacity: 0.8 !important; }
        </style>
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            new MutationObserver(function() {
                var el = document.getElementById('scrip-input');
                if (!el) return;
                var inputs = el.querySelectorAll('input');
                inputs.forEach(function(inp) {
                    var node = inp.parentElement;
                    var inMenu = false;
                    while (node && node !== el) {
                        var cn = (typeof node.className === 'string') ? node.className : '';
                        if (cn.indexOf('menu') > -1 && cn.indexOf('List') === -1) {
                            inMenu = true; break;
                        }
                        node = node.parentElement;
                    }
                    if (inMenu) {
                        var w = inp.closest('div');
                        while (w && w !== el) {
                            var pcn = w.parentElement ? ((typeof w.parentElement.className === 'string') ? w.parentElement.className : '') : '';
                            if (pcn.indexOf('menu') > -1 && pcn.indexOf('List') === -1) {
                                w.style.display = 'none';
                                break;
                            }
                            w = w.parentElement;
                        }
                    }
                });
            }).observe(document.body, {childList: true, subtree: true});
        });
        </script>
    </head>
    <body>
        {%app_entry%}
        {%config%}
        {%scripts%}
        {%renderer%}
    </body>
</html>'''

SIDEBAR_STYLE = {}  # Managed by CSS class .sidebar-dark
CONTENT_STYLE = {"marginLeft": "240px", "marginRight": "20px", "padding": "24px 20px", "minHeight": "100vh"}

# --- LAYOUT DEFINITIONS ---

def layout_iv_pro_view():
    return html.Div([
        # --- HEADER ---
        html.Div([
            html.Div([
                html.Div([
                    html.Span("IV", style={'color': '#00e676', 'fontWeight': '800'}),
                    html.Span(" PRO", style={'color': '#fff', 'fontWeight': '300'}),
                ], style={'fontSize': '24px', 'letterSpacing': '2px'}),
                html.Div("Live Analyzer · Smart Filter", style={
                    'color': 'rgba(255,255,255,0.5)', 'fontSize': '11px', 
                    'letterSpacing': '3px', 'textTransform': 'uppercase', 'marginTop': '2px'
                }),
            ], style={'display': 'inline-block'}),
            html.Div(id='ivpro-clock', style={
                'float': 'right', 'color': '#00e676', 'marginTop': '12px',
                'fontSize': '14px', 'fontWeight': '500', 'letterSpacing': '1px'
            })
        ], style={
            'background': 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
            'padding': '16px 28px', 'borderBottom': '1px solid rgba(0,230,118,0.15)'
        }),
        
        # --- HORIZONTAL CONTROLS BAR ---
        html.Div([
            # Symbol
            html.Div([
                html.Label("Symbol", className="ctrl-label"),
                dcc.Dropdown(
                    id='ivpro-dd-symbol',
                    options=[{'label': s, 'value': s} for s in sorted(base_name_meta_lookup.keys())],
                    value='NIFTY',
                    clearable=False,
                    className='ctrl-dropdown'
                ),
            ], className='ctrl-group', style={'flex': '1.2'}),
            
            # Expiry
            html.Div([
                html.Label("Expiry", className="ctrl-label"),
                dcc.Dropdown(
                    id='ivpro-dd-expiry',
                    placeholder="Select Expiry",
                    clearable=False,
                    className='ctrl-dropdown'
                ),
            ], className='ctrl-group', style={'flex': '1'}),
            
            # Strike
            html.Div([
                html.Label("Strike", className="ctrl-label"),
                dcc.Dropdown(
                    id='ivpro-dd-strike',
                    placeholder="Select Strike",
                    className='ctrl-dropdown'
                ),
            ], className='ctrl-group', style={'flex': '1'}),
            
            # Timeframe Buttons
            html.Div([
                html.Label("Timeframe", className="ctrl-label"),
                html.Div([
                    html.Button("1D", id="ivpro-btn-1d", n_clicks=0, className="time-btn"),
                    html.Button("5D", id="ivpro-btn-5d", n_clicks=0, className="time-btn"),
                    html.Button("1M", id="ivpro-btn-1m", n_clicks=0, className="time-btn"),
                    html.Button("All", id="ivpro-btn-all", n_clicks=0, className="time-btn"),
                ], style={'display': 'flex', 'gap': '4px'}),
            ], className='ctrl-group', style={'flex': '0.8'}),
            
        ], style={
            'display': 'flex', 'gap': '16px', 'alignItems': 'flex-end',
            'padding': '16px 28px',
            'background': 'rgba(255,255,255,0.03)',
            'borderBottom': '1px solid rgba(255,255,255,0.06)',
            'backdropFilter': 'blur(10px)',
            'position': 'relative', 'zIndex': 10,
        }),
        
        # --- CHARTS (Full Width) ---
        html.Div([
            html.Div([
                dcc.Graph(id='ivpro-ce-chart', style={'height': '42vh'}, config={'scrollZoom': True}),
            ], className='chart-card'),
            html.Div([
                dcc.Graph(id='ivpro-pe-chart', style={'height': '42vh'}, config={'scrollZoom': True}),
            ], className='chart-card'),
        ], style={'padding': '20px 28px', 'display': 'flex', 'flexDirection': 'column', 'gap': '16px'}),
        
        # Store selected timeframe
        dcc.Store(id='ivpro-store-tf', data='1D'),
        dcc.Interval(id='ivpro-interval', interval=60*1000, n_intervals=0), 
    ], className='ivpro-page', style={
        'backgroundColor': '#0a0a14', 'minHeight': '100vh', 
        'color': '#eee', 'fontFamily': "'Inter', 'Segoe UI', sans-serif",
        'margin': '-2rem -1rem', 'padding': '0' # Counteract default padding
    })

def layout_market_overview():
    curr_user = "Unknown"
    if 'kite' in globals():
        try:
            profile = kite.profile()
            curr_user = profile.get('user_name', 'Unknown')
        except:
             pass
    return html.Div([
        # --- HERO BANNER ---
        html.Div([
            html.H2([
                "Welcome, ",
                html.Strong(curr_user),
            ], className='hero-title'),
            html.P("Options Analytics · Real-Time IV · Smart Filtering", className='hero-sub'),
        ], className='hero-banner'),

        # --- FEATURE CARDS GRID ---
        html.Div([
            dcc.Link([
                html.Span("📊", className='feature-icon'),
                html.Div("Vol Analyzer", className='feature-name'),
                html.Div("Real-time IV smile & option chain analysis", className='feature-desc'),
            ], href='/analyzer', className='feature-card'),
            dcc.Link([
                html.Span("🔥", className='feature-icon'),
                html.Div("IV Heatmap", className='feature-name'),
                html.Div("Market-wide IV skew visualization", className='feature-desc'),
            ], href='/', className='feature-card'),
            dcc.Link([
                html.Span("⚖️", className='feature-icon'),
                html.Div("PCR Skew", className='feature-name'),
                html.Div("Put-Call OI ratio across strikes", className='feature-desc'),
            ], href='/pcr', className='feature-card'),
            dcc.Link([
                html.Span("📈", className='feature-icon'),
                html.Div("IV Ratio", className='feature-name'),
                html.Div("Put/Call IV ratio heatmap", className='feature-desc'),
            ], href='/iv-ratio', className='feature-card'),
            dcc.Link([
                html.Span("🎯", className='feature-icon'),
                html.Div("Max Pain", className='feature-name'),
                html.Div("Expiry pain analysis & drill-down", className='feature-desc'),
            ], href='/max-pain', className='feature-card'),
            dcc.Link([
                html.Span("📅", className='feature-icon'),
                html.Div("Calendar IV", className='feature-name'),
                html.Div("Near vs far month spread scanner", className='feature-desc'),
            ], href='/calendar', className='feature-card'),
        ], className='feature-grid'),

        # --- IV HEATMAP SECTION ---
        html.Div([
            html.H3("Market-Wide IV Skew Heatmap", className='page-title', style={'fontSize': '20px'}),
            html.P("IV Skew (IV / ATM IV) across all NFO symbols", className='page-subtitle', style={'marginBottom': '20px'}),
            html.Div([
                html.Button("Generate Heatmap", id="btn-heatmap", n_clicks=0, className='btn-glow'),
                html.Span("Takes ~1 min for full scan", className='btn-hint'),
            ], style={'marginBottom': '20px'}),
            dcc.Loading(
                id="loading-heatmap",
                children=[dcc.Graph(id="graph-heatmap", style={'height': '800px'})],
                type="cube", color="#00e676"
            )
        ], className='dark-card')
    ])

def layout_pcr_view():
    return html.Div([
        html.Div([
            html.H2("PCR Skew Heatmap", className='page-title'),
            html.P("Put-Call OI Ratio (PE OI / CE OI) across NFO symbols by moneyness", className='page-subtitle'),
        ], style={'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.Button("Generate PCR Heatmap", id="btn-pcr-heatmap", n_clicks=0, className='btn-glow-blue'),
                html.Span("Uses cached data from IV Heatmap if available", className='btn-hint'),
            ], style={'marginBottom': '20px'}),
            dcc.Loading(
                id="loading-pcr-heatmap",
                children=[dcc.Graph(id="graph-pcr-heatmap", style={'height': '800px'})],
                type="cube", color="#2979ff"
            )
        ], className='dark-card')
    ])

def layout_iv_ratio_view():
    return html.Div([
        html.Div([
            html.H2("Put/Call IV Ratio Heatmap", className='page-title'),
            html.P("Put IV / Call IV ratio across NFO symbols by moneyness", className='page-subtitle'),
        ], style={'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.Button("Generate IV Ratio Heatmap", id="btn-ivratio-heatmap", n_clicks=0, className='btn-glow-purple'),
                html.Span("Uses cached data from IV Heatmap if available", className='btn-hint'),
            ], style={'marginBottom': '20px'}),
            dcc.Loading(
                id="loading-ivratio-heatmap",
                children=[dcc.Graph(id="graph-ivratio-heatmap", style={'height': '800px'})],
                type="cube", color="#7c4dff"
            )
        ], className='dark-card')
    ])

def layout_max_pain_view():
    return html.Div([
        html.Div([
            html.H2("Max Pain Analysis", className='page-title'),
            html.P("Strike-level pain analysis for option writers", className='page-subtitle'),
        ], style={'marginBottom': '20px'}),
        html.Div([
            html.Button("Generate Max Pain", id="btn-maxpain", n_clicks=0, className='btn-glow-purple',
                         style={'marginBottom': '20px'}),
            dcc.Loading([
                html.Div(id='maxpain-table-container', style={'margin': '20px 0'}),
                dcc.Graph(id='graph-maxpain-heatmap', style={'height': '900px'}),
                html.H3(id='drilldown-title', style={'textAlign': 'center', 'marginTop': '30px', 'color': 'rgba(255,255,255,0.5)'}),
                dcc.Graph(id='graph-maxpain-drilldown', style={'height': '500px'}),
            ], color="#7c4dff"),
        ], className='dark-card'),
        html.Div([
            html.H4("How to Interpret Max Pain"),
            html.Ul([
                html.Li("Max Pain = strike where total option buyer losses are maximized (option writers profit most)."),
                html.Li("If Spot > Max Pain → Bearish bias (market may pull back toward Max Pain)."),
                html.Li("If Spot < Max Pain → Bullish bias (market may rally toward Max Pain)."),
                html.Li("High pain concentration at one strike = strong magnet for expiry settlement."),
                html.Li("Click a row in the table to see the full pain profile for that symbol."),
            ])
        ], className='info-box')
    ])

def layout_calendar_view():
    return html.Div([
        html.Div([
            html.H2("Calendar IV Spread", className='page-title'),
            html.P("Near-month IV vs Next-month IV — find calendar spread opportunities", className='page-subtitle'),
        ], style={'marginBottom': '20px'}),
        html.Div([
            html.Button("Generate Calendar Spread", id="btn-calendar", n_clicks=0, className='btn-glow-orange',
                         style={'marginBottom': '20px'}),
            dcc.Loading([
                dcc.Graph(id='graph-calendar-atm', style={'height': '900px'}),
                html.Div(id='calendar-table-container', style={'margin': '20px 0'}),
                html.H3(id='calendar-drilldown-title', style={'textAlign': 'center', 'marginTop': '30px', 'color': 'rgba(255,255,255,0.5)'}),
                dcc.Graph(id='graph-calendar-drilldown', style={'height': '550px'}),
            ], color="#ff6d00"),
            dcc.Store(id='store-calendar-data', storage_type='session'),
        ], className='dark-card'),
        html.Div([
            html.H4("How to Read Calendar Spreads"),
            html.Ul([
                html.Li("Positive spread (Near IV > Far IV) = Contango (normal). Sell near, buy far."),
                html.Li("Negative spread (Near IV < Far IV) = Backwardation (unusual). Buy near, sell far — or investigate."),
                html.Li("Large |spread| = potential calendar spread trade opportunity."),
                html.Li("Click a row in the table to see the full strike-by-strike IV comparison."),
                html.Li("Green bars = Far IV higher at that strike (backwardation). Red = Near IV higher (contango)."),
            ])
        ], className='info-box')
    ])

def layout_analyzer_view():
    return html.Div(children=[
        html.H1("Real-Time Option Analyzer", className='page-title', style={'textAlign': 'center', 'marginBottom': '20px'}),
        # --- Top Controls ---
        html.Div([
            html.Label("Symbol", style={'fontWeight': '600', 'marginRight': '8px', 'color': 'rgba(255,255,255,0.6)', 'fontSize': '13px', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
            dcc.Dropdown(
                id='scrip-input',
                options=[{'label': k, 'value': k} for k in sorted(base_name_meta_lookup.keys())],
                value='NIFTY', placeholder='Search symbol...', searchable=True, clearable=False,
                style={'width': '220px', 'display': 'inline-block'}
            ),
            html.Button('Analyze Snapshot', id='submit-btn', n_clicks=0, className='btn-glow-blue'),
            html.Div([
                html.Label("Expiry", style={'fontWeight': '600', 'marginRight': '8px', 'color': 'rgba(255,255,255,0.6)', 'fontSize': '13px', 'textTransform': 'uppercase', 'letterSpacing': '0.5px'}),
                dcc.Dropdown(id='expiry-dropdown', style={'width': '200px', 'display': 'inline-block'})
            ], style={'display': 'inline-flex', 'alignItems': 'center'})
        ], className='analyzer-controls'),
        # --- Section 1: Snapshot Smile ---
        dcc.Loading(
            type="dot", color="#00e676",
            children=[
                dcc.Graph(id='smile-graph', style={'height': '600px'}),
                html.Div(id='click-data-output', style={'padding': '20px', 'textAlign': 'center'}),
                html.Div(id='error-msg', style={'color': '#ff5252', 'textAlign': 'center', 'marginTop': '20px'})
            ]
        ),
        html.Hr(style={'margin': '30px 0', 'borderColor': 'rgba(255,255,255,0.06)'}),
        # --- Section 2: LIVE IV MONITOR ---
        html.Div([
            html.H3("⏱️ Live 1-Minute IV Tracker", className='page-title', style={'textAlign': 'center', 'fontSize': '20px', 'marginBottom': '16px'}),
            html.Div([
                html.Label("Select Strike to Track:", style={'fontWeight': '600', 'color': 'rgba(255,255,255,0.6)', 'fontSize': '13px'}),
                dcc.Dropdown(
                    id='history-strike-dropdown',
                    placeholder="Select a strike...",
                    style={'width': '200px', 'margin': '0 auto'}
                ),
                html.Div(id='monitor-status', style={'marginTop': '5px', 'fontSize': '12px', 'color': '#00e676'})
            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
            dcc.Graph(id='iv-history-graph', style={'height': '500px'}),
            dcc.Interval(
                id='monitor-interval',
                interval=60 * 1000,
                n_intervals=0
            )
        ], className='dark-card')
    ])

# --- MAIN APP LAYOUT ---
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id='sidebar-state', data='open'),
    dcc.Store(id='store-heatmap-fig', storage_type='session'),
    dcc.Store(id='store-pcr-fig', storage_type='session'),
    dcc.Store(id='store-ivratio-fig', storage_type='session'),
    dcc.Store(id='store-maxpain-data', storage_type='session'),
    dcc.Store(id='store-calendar-data', storage_type='session'),

    html.Button("☰", id="sidebar-toggle", n_clicks=0, style={
        'position': 'fixed', 'top': '14px', 'left': '14px', 'zIndex': 1100,
        'fontSize': '18px', 'padding': '6px 12px', 'cursor': 'pointer',
        'background': 'rgba(255,255,255,0.06)', 'color': '#00e676', 'border': '1px solid rgba(255,255,255,0.1)',
        'borderRadius': '8px', 'backdropFilter': 'blur(8px)', 'transition': 'all 0.2s'
    }),

    html.Div([
        # Sidebar Logo
        html.Div([
            html.Div([
                html.Span("Kite", style={'color': '#00e676', 'fontWeight': '700'}),
                html.Span(" Analytics", style={'color': '#fff', 'fontWeight': '300'}),
            ], className='sidebar-logo-text'),
            html.Div("Options Intelligence", className='sidebar-logo-sub'),
        ], className='sidebar-logo', style={'marginTop': '36px'}),

        # Navigation Links
        dcc.Link([html.Span("🏠", className='nav-icon'), "Home"], href='/', className='nav-link'),
        dcc.Link([html.Span("📊", className='nav-icon'), "Vol Analyzer"], href='/analyzer', className='nav-link'),
        dcc.Link([html.Span("⚖️", className='nav-icon'), "PCR Skew"], href='/pcr', className='nav-link'),
        dcc.Link([html.Span("📈", className='nav-icon'), "IV Ratio"], href='/iv-ratio', className='nav-link'),
        dcc.Link([html.Span("🎯", className='nav-icon'), "Max Pain"], href='/max-pain', className='nav-link'),
        dcc.Link([html.Span("📅", className='nav-icon'), "Calendar IV"], href='/calendar', className='nav-link'),

        # Separator
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.06)', 'margin': '8px 20px'}),

        # IV Pro (special highlight)
        dcc.Link([
            html.Span("⚡", className='nav-icon'),
            html.Span("IV Pro", style={'color': '#00e676', 'fontWeight': '700'}),
        ], href='/iv-pro', className='nav-link', style={'borderLeft': '3px solid #00e676'}),
    ], id='sidebar', className='sidebar-dark'),

    html.Div(id="page-content", style=CONTENT_STYLE)
])

# ==============================================================================
# 5. CALLBACKS
# ==============================================================================

# --- Navigation ---
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/" or pathname == "/home":
        return layout_market_overview()
    elif pathname == "/analyzer":
        return layout_analyzer_view()
    elif pathname == "/pcr":
        return layout_pcr_view()
    elif pathname == "/iv-ratio":
        return layout_iv_ratio_view()
    elif pathname == "/max-pain":
        return layout_max_pain_view()
    elif pathname == "/calendar":
        return layout_calendar_view()
    elif pathname == "/iv-pro":
        return layout_iv_pro_view()
    else:
        return html.Div([
            html.H1("404: Not found", style={'color': '#ff5252'}),
            html.Hr(style={'borderColor': 'rgba(255,255,255,0.06)'}),
            html.P(f"The pathname {pathname} was not recognised..."),
        ], className='dark-card')

# --- Sidebar Toggle ---
app.clientside_callback(
    """
    function(n_clicks, current_state) {
        if (!n_clicks) return [current_state, {}, {}];
        var new_state = current_state === 'open' ? 'closed' : 'open';
        var sidebar_style, content_style;
        if (new_state === 'closed') {
            sidebar_style = {'position':'fixed','top':0,'left':0,'bottom':0,'width':'0','padding':'0','overflow':'hidden','background':'linear-gradient(180deg, #0d0d1a 0%, #111128 100%)','borderRight':'1px solid rgba(255,255,255,0.06)','transition':'all 0.3s ease','zIndex':1000};
            content_style = {'marginLeft':'20px','marginRight':'20px','padding':'24px 20px','minHeight':'100vh','transition':'all 0.3s ease'};
        } else {
            sidebar_style = {'position':'fixed','top':0,'left':0,'bottom':0,'width':'220px','padding':'20px 0','background':'linear-gradient(180deg, #0d0d1a 0%, #111128 100%)','borderRight':'1px solid rgba(255,255,255,0.06)','transition':'all 0.3s ease','overflowY':'auto','zIndex':1000};
            content_style = {'marginLeft':'240px','marginRight':'20px','padding':'24px 20px','minHeight':'100vh','transition':'all 0.3s ease'};
        }
        return [new_state, sidebar_style, content_style];
    }
    """,
    [Output('sidebar-state', 'data'), Output('sidebar', 'style'), Output('page-content', 'style')],
    [Input('sidebar-toggle', 'n_clicks')],
    [State('sidebar-state', 'data')]
)

# --- ANALYZER: Snapshot Callback ---
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
    final_expiry = None
    final_options = existing_options if existing_options else []
    if trigger_id == 'submit-btn' or not final_options:
        if scrip_name in base_name_meta_lookup:
            exps = base_name_meta_lookup[scrip_name].get('options_expiries', [])
            if exps:
                final_options = [{'label': pd.to_datetime(d).strftime('%Y-%b-%d').upper(), 'value': str(d)} for d in exps]
                final_expiry = final_options[0]['value']
            else:
                final_options = []
                final_expiry = None
        else:
            final_options = []
            final_expiry = None
    elif trigger_id == 'expiry-dropdown':
        final_expiry = expiry_value
    if not final_expiry and final_options:
         final_expiry = final_options[0]['value']
    try:
        df, spot_price, atm_iv = get_option_data(scrip_name, expiry=final_expiry)
        if df.empty:
            return go.Figure(), f"No data found for {scrip_name}. Check spelling or market status.", final_options, final_expiry

        # --- MCX Bid-Ask Spread Filter for cleaner IV Smile ---
        scrip_exchange = base_name_meta_lookup.get(scrip_name, {}).get('exchange', '')
        if scrip_exchange == 'MCX' and len(df) >= 2:
            strikes_sorted = df['STRIKE'].dropna().sort_values().values
            diffs = np.diff(strikes_sorted)
            strike_gap = np.median(diffs) if len(diffs) > 0 else 0
            threshold = 2 * strike_gap
            if threshold > 0:
                # Independently filter CE and PE
                ce_spread = (df['CE-ASK'].fillna(0) - df['CE-BID'].fillna(0)).abs()
                pe_spread = (df['PE-ASK'].fillna(0) - df['PE-BID'].fillna(0)).abs()
                df.loc[ce_spread > threshold, 'CE-IV'] = np.nan
                df.loc[pe_spread > threshold, 'PE-IV'] = np.nan

        # --- Deep ITM Filter: hide strikes > 20% ITM ---
        if spot_price > 0:
            itm_lo = spot_price * 0.80  # CE is deep ITM below this
            itm_hi = spot_price * 1.20  # PE is deep ITM above this
            df.loc[df['STRIKE'] < itm_lo, 'CE-IV'] = np.nan
            df.loc[df['STRIKE'] > itm_hi, 'PE-IV'] = np.nan
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['STRIKE'], y=df['CE-OI'], name='CE OI',
            marker=dict(color='rgba(220, 53, 69, 0.45)'), yaxis='y2'))
        fig.add_trace(go.Bar(x=df['STRIKE'], y=df['PE-OI'], name='PE OI',
            marker=dict(color='rgba(40, 167, 69, 0.45)'), yaxis='y2'))
        fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['CE-IV'], mode='lines+markers', name='CE IV',
            line=dict(color='#28a745', width=3), marker=dict(size=6),
            connectgaps=True,
            customdata=df[['CE-LTP', 'CE-BID', 'CE-ASK', 'CE-OI']].values,
            hovertemplate="<b>Strike: %{x}</b><br>IV: %{y:.2f}%<br>LTP: %{customdata[0]}<br>Bid: %{customdata[1]}<br>Ask: %{customdata[2]}<br>OI: %{customdata[3]}<extra></extra>"))
        fig.add_trace(go.Scatter(x=df['STRIKE'], y=df['PE-IV'], mode='lines+markers', name='PE IV',
            line=dict(color='#dc3545', width=3), marker=dict(size=6),
            connectgaps=True,
            customdata=df[['PE-LTP', 'PE-BID', 'PE-ASK', 'PE-OI']].values,
            hovertemplate="<b>Strike: %{x}</b><br>IV: %{y:.2f}%<br>LTP: %{customdata[0]}<br>Bid: %{customdata[1]}<br>Ask: %{customdata[2]}<br>OI: %{customdata[3]}<extra></extra>"))
        fig.update_layout(
            title=f"Volatility Smile: {scrip_name}",
            xaxis_title="Strike Price", yaxis_title="Implied Volatility (%)",
            yaxis2=dict(title="Open Interest", overlaying="y", side="right", showgrid=False,
                        range=[0, df[['CE-OI', 'PE-OI']].max().max() * 3]),
            template="plotly_white", hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        if spot_price > 0:
            anno_text = f"Spot: {spot_price}"
            if atm_iv > 0:
                anno_text += f" | ATM IV: {atm_iv:.2f}%"
            fig.add_vline(x=spot_price, line_width=2, line_dash="dash", line_color="black",
                annotation_text=anno_text, annotation_position="top right")
        return fig, "", final_options, final_expiry
    except Exception as e:
        return go.Figure(), f"An error occurred: {str(e)}", final_options, final_expiry

# --- Click Data Display ---
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
        custom_data = point.get('customdata', [0, 0, 0, 0])
        ltp, bid, ask, oi = custom_data[0], custom_data[1], custom_data[2], custom_data[3]
        return html.Div([
            html.H3(f"Selected Strike: {strike}", style={'margin': '0 0 10px 0', 'color': '#333'}),
            html.Div([
                html.Div([html.Strong("IV:"), html.Span(f" {iv:.2f}%")], style={'display': 'inline-block', 'margin': '0 10px'}),
                html.Div([html.Strong("LTP:"), html.Span(f" {ltp}")], style={'display': 'inline-block', 'margin': '0 10px'}),
                html.Div([html.Strong("Bid:"), html.Span(f" {bid}")], style={'display': 'inline-block', 'margin': '0 10px'}),
                html.Div([html.Strong("Ask:"), html.Span(f" {ask}")], style={'display': 'inline-block', 'margin': '0 10px'}),
                html.Div([html.Strong("OI:"), html.Span(f" {oi}")], style={'display': 'inline-block', 'margin': '0 10px'}),
            ], style={'backgroundColor': '#fff', 'padding': '15px', 'borderRadius': '8px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'display': 'inline-block'})
        ])
    except Exception as e:
        return html.Div(f"Error displaying details: {str(e)}")

# --- IV Heatmap Callback ---
@app.callback(
    [Output("graph-heatmap", "figure"),
     Output("store-heatmap-fig", "data")],
    [Input("btn-heatmap", "n_clicks")],
    [State("store-heatmap-fig", "data")]
)
def update_heatmap(n_clicks, stored_fig):
    if not n_clicks:
        if stored_fig:
            return go.Figure(stored_fig), dash.no_update
        return go.Figure(), dash.no_update
    df_heat = _get_heatmap_data()
    if df_heat.empty:
        return go.Figure(), dash.no_update
    if 'Expiry_Rank' in df_heat.columns:
        df_heat = df_heat[df_heat['Expiry_Rank'] == 1].copy()
    bin_step = 0.02
    df_heat['Moneyness_Bin'] = (df_heat['Moneyness'] / bin_step).round() * bin_step
    z_mid = 1.0
    colors = 'RdBu_r'
    title = "IV / ATM IV"
    def create_pivot(df_sub):
        p_ratio = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='Ratio', aggfunc='mean').sort_index(ascending=False)
        p_oi = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='OI', aggfunc='mean').sort_index(ascending=False)
        p_bid = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='Bid', aggfunc='mean').sort_index(ascending=False)
        p_ask = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='Ask', aggfunc='mean').sort_index(ascending=False)
        p_strike = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='Strike', aggfunc='mean').sort_index(ascending=False)
        p_iv = df_sub.pivot_table(index='Symbol', columns='Moneyness_Bin', values='IV', aggfunc='mean').sort_index(ascending=False)
        cols = [c for c in p_ratio.columns if 0.8 <= c <= 1.2]
        return p_ratio[cols], p_oi[cols], p_bid[cols], p_ask[cols], p_strike[cols], p_iv[cols]
    df_ce = df_heat[df_heat['Type'] == 'CE']
    ce_ratio, ce_oi, ce_bid, ce_ask, ce_strike, ce_iv = create_pivot(df_ce)
    ce_custom = np.dstack((ce_oi.values, ce_bid.values, ce_ask.values, ce_strike.values, ce_iv.values))
    df_pe = df_heat[df_heat['Type'] == 'PE']
    pe_ratio, pe_oi, pe_bid, pe_ask, pe_strike, pe_iv = create_pivot(df_pe)
    pe_custom = np.dstack((pe_oi.values, pe_bid.values, pe_ask.values, pe_strike.values, pe_iv.values))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=(f"CE {title}", f"PE {title}"))
    hm_args = dict(colorscale=colors, zmid=z_mid, colorbar=dict(title=title, len=0.45))
    fig.add_trace(go.Heatmap(z=ce_ratio.values, x=ce_ratio.columns, y=ce_ratio.index,
        customdata=ce_custom, colorbar_y=0.8,
        hovertemplate="<b>Symbol: %{y}</b><br>Moneyness: %{x}<br>Strike: %{customdata[3]:,.0f}<br>IV: %{customdata[4]:.2f}%<br>Ratio: %{z:.2f}<br>OI: %{customdata[0]:,}<br>Bid: %{customdata[1]:.2f}<br>Ask: %{customdata[2]:.2f}<extra>CE</extra>",
        **hm_args), row=1, col=1)
    fig.add_trace(go.Heatmap(z=pe_ratio.values, x=pe_ratio.columns, y=pe_ratio.index,
        customdata=pe_custom, colorbar_y=0.2,
        hovertemplate="<b>Symbol: %{y}</b><br>Moneyness: %{x}<br>Strike: %{customdata[3]:,.0f}<br>IV: %{customdata[4]:.2f}%<br>Ratio: %{z:.2f}<br>OI: %{customdata[0]:,}<br>Bid: %{customdata[1]:.2f}<br>Ask: %{customdata[2]:.2f}<extra>PE</extra>",
        **hm_args), row=2, col=1)
    fig.update_layout(
        title=f"Market IV Skew Heatmap (All {len(ce_ratio)} Symbols — NFO + MCX + BFO)",
        height=max(800, len(ce_ratio) * 25),
        xaxis2={'title': "Moneyness (Strike / Spot)", 'tickmode': 'linear', 'dtick': 0.05})
    return fig, fig

# --- PCR HEATMAP CALLBACK ---
@app.callback(
    [Output("graph-pcr-heatmap", "figure"),
     Output("store-pcr-fig", "data")],
    [Input("btn-pcr-heatmap", "n_clicks")],
    [State("store-pcr-fig", "data")]
)
def update_pcr_heatmap(n_clicks, stored_fig):
    if not n_clicks:
        if stored_fig:
            return go.Figure(stored_fig), dash.no_update
        return go.Figure(), dash.no_update
    df_heat = _get_heatmap_data()
    if df_heat.empty:
        return go.Figure(), dash.no_update
    if 'Expiry_Rank' in df_heat.columns:
        df_heat = df_heat[df_heat['Expiry_Rank'] == 1].copy()
    bin_step = 0.02
    df_heat = df_heat.copy()
    df_heat['Moneyness_Bin'] = (df_heat['Moneyness'] / bin_step).round() * bin_step
    df_heat = df_heat[(df_heat['Moneyness_Bin'] >= 0.8) & (df_heat['Moneyness_Bin'] <= 1.2)]
    df_ce = df_heat[df_heat['Type'] == 'CE']
    df_pe = df_heat[df_heat['Type'] == 'PE']
    pivot_ce_oi = df_ce.pivot_table(index='Symbol', columns='Moneyness_Bin', values='OI', aggfunc='sum').sort_index(ascending=False)
    pivot_pe_oi = df_pe.pivot_table(index='Symbol', columns='Moneyness_Bin', values='OI', aggfunc='sum').sort_index(ascending=False)
    all_symbols = sorted(set(pivot_ce_oi.index) | set(pivot_pe_oi.index), reverse=True)
    all_bins = sorted(set(list(pivot_ce_oi.columns) + list(pivot_pe_oi.columns)))
    pivot_ce_oi = pivot_ce_oi.reindex(index=all_symbols, columns=all_bins, fill_value=0)
    pivot_pe_oi = pivot_pe_oi.reindex(index=all_symbols, columns=all_bins, fill_value=0)
    pcr = pivot_pe_oi / pivot_ce_oi.replace(0, np.nan)
    custom = np.dstack((pivot_ce_oi.values, pivot_pe_oi.values))
    fig = go.Figure(go.Heatmap(z=pcr.values, x=pcr.columns, y=pcr.index, customdata=custom,
        colorscale='RdBu', zmid=1.0, zmin=0, zmax=3.0, colorbar=dict(title="PCR"),
        hovertemplate="<b>Symbol: %{y}</b><br>Moneyness: %{x}<br>PCR: %{z:.2f}<br>CE OI: %{customdata[0]:,}<br>PE OI: %{customdata[1]:,}<extra></extra>"))
    fig.update_layout(title=f"NFO Put-Call Ratio Heatmap ({len(all_symbols)} Symbols)",
        height=max(800, len(all_symbols) * 25),
        xaxis={'title': 'Moneyness (Strike / Spot)', 'tickmode': 'linear', 'dtick': 0.05})
    return fig, fig

# --- PUT/CALL IV RATIO HEATMAP CALLBACK ---
@app.callback(
    [Output("graph-ivratio-heatmap", "figure"),
     Output("store-ivratio-fig", "data")],
    [Input("btn-ivratio-heatmap", "n_clicks")],
    [State("store-ivratio-fig", "data")]
)
def update_ivratio_heatmap(n_clicks, stored_fig):
    if not n_clicks:
        if stored_fig:
            return go.Figure(stored_fig), dash.no_update
        return go.Figure(), dash.no_update
    df_heat = _get_heatmap_data()
    if df_heat.empty:
        return go.Figure(), dash.no_update
    if 'Expiry_Rank' in df_heat.columns:
        df_heat = df_heat[df_heat['Expiry_Rank'] == 1].copy()
    bin_step = 0.02
    df_heat = df_heat.copy()
    df_heat['Moneyness_Bin'] = (df_heat['Moneyness'] / bin_step).round() * bin_step
    df_heat = df_heat[(df_heat['Moneyness_Bin'] >= 0.8) & (df_heat['Moneyness_Bin'] <= 1.2)]
    df_ce = df_heat[df_heat['Type'] == 'CE']
    df_pe = df_heat[df_heat['Type'] == 'PE']
    pivot_ce_iv = df_ce.pivot_table(index='Symbol', columns='Moneyness_Bin', values='IV', aggfunc='mean').sort_index(ascending=False)
    pivot_pe_iv = df_pe.pivot_table(index='Symbol', columns='Moneyness_Bin', values='IV', aggfunc='mean').sort_index(ascending=False)
    all_symbols = sorted(set(pivot_ce_iv.index) | set(pivot_pe_iv.index), reverse=True)
    all_bins = sorted(set(list(pivot_ce_iv.columns) + list(pivot_pe_iv.columns)))
    pivot_ce_iv = pivot_ce_iv.reindex(index=all_symbols, columns=all_bins)
    pivot_pe_iv = pivot_pe_iv.reindex(index=all_symbols, columns=all_bins)
    iv_ratio = pivot_pe_iv / pivot_ce_iv.replace(0, np.nan)
    custom = np.dstack((pivot_ce_iv.values, pivot_pe_iv.values))
    fig = go.Figure(go.Heatmap(z=iv_ratio.values, x=iv_ratio.columns, y=iv_ratio.index, customdata=custom,
        colorscale='RdBu', zmid=1.0, zmin=0.5, zmax=1.5, colorbar=dict(title="PE/CE IV"),
        hovertemplate="<b>Symbol: %{y}</b><br>Moneyness: %{x}<br>PE/CE IV: %{z:.2f}<br>CE IV: %{customdata[0]:.2f}%<br>PE IV: %{customdata[1]:.2f}%<extra></extra>"))
    fig.update_layout(title=f"NFO Put/Call IV Ratio Heatmap ({len(all_symbols)} Symbols)",
        height=max(800, len(all_symbols) * 25),
        xaxis={'title': 'Moneyness (Strike / Spot)', 'tickmode': 'linear', 'dtick': 0.05})
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
            summary = pd.DataFrame(stored_data['summary'])
            pain_profiles = {k: pd.DataFrame(v) for k, v in stored_data['profiles'].items()}
        else:
            return html.Div(), go.Figure(), dash.no_update
    else:
        df_heat = _get_heatmap_data()
        if df_heat.empty:
            return html.Div("No data available."), go.Figure(), dash.no_update
        if 'Expiry_Rank' in df_heat.columns:
            df_heat = df_heat[df_heat['Expiry_Rank'] == 1].copy()
        summary, pain_profiles = compute_max_pain(df_heat)
    if summary.empty:
        return html.Div("No Max Pain data computed."), go.Figure(), dash.no_update
    table_df = summary[['Symbol', 'Spot', 'Max Pain', 'Distance%', 'Bias']].copy()
    table = dash_table.DataTable(
        id='maxpain-table',
        columns=[{'name': c, 'id': c} for c in table_df.columns],
        data=table_df.to_dict('records'),
        sort_action='native', filter_action='native', page_size=20, row_selectable='single',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': '14px'},
        style_header={'backgroundColor': '#007bff', 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'filter_query': '{Bias} contains "Bull"'}, 'backgroundColor': '#d4edda', 'color': '#155724'},
            {'if': {'filter_query': '{Bias} contains "Bear"'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'},
        ])
    symbols = summary['Symbol'].tolist()
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
    z_normalized = []
    for z_row in z_matrix:
        row_max = max(z_row) if max(z_row) > 0 else 1
        z_normalized.append([v / row_max for v in z_row])
    fig_heatmap = go.Figure(go.Heatmap(z=z_normalized, x=x_labels, y=symbols,
        customdata=custom_matrix, colorscale='YlOrRd', zmin=0, zmax=1,
        colorbar=dict(title='Pain (normalized)', tickvals=[0, 0.5, 1], ticktext=['Low', 'Mid', 'High']),
        hovertemplate="<b>%{y}</b><br>Rank: %{x}<br>Strike: %{customdata[0]}<br>Total Pain: %{customdata[1]:,.0f}<br>CE Pain: %{customdata[2]:,.0f}<br>PE Pain: %{customdata[3]:,.0f}<br>CE OI: %{customdata[4]:,}<br>PE OI: %{customdata[5]:,}<extra></extra>"))
    fig_heatmap.update_layout(
        title=f"Top 5 Highest-Pain Strikes ({len(symbols)} Symbols) — Color normalized per symbol",
        height=max(800, len(symbols) * 25), yaxis={'dtick': 1})
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
    profiles = stored_data.get('profiles', {})
    if sym not in profiles:
        return go.Figure(), f"No profile data for {sym}"
    profile = pd.DataFrame(profiles[sym])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=profile['Strike'], y=profile['CE_Pain'], name='CE Pain (Resistance)',
        marker_color='#dc3545', hovertemplate="Strike: %{x}<br>CE Pain: %{y:,.0f}<extra>CE</extra>"))
    fig.add_trace(go.Bar(x=profile['Strike'], y=profile['PE_Pain'], name='PE Pain (Support)',
        marker_color='#28a745', hovertemplate="Strike: %{x}<br>PE Pain: %{y:,.0f}<extra>PE</extra>"))
    fig.add_vline(x=max_pain, line_width=3, line_dash="dash", line_color="blue",
        annotation_text=f"Max Pain: {max_pain}", annotation_position="top right")
    fig.add_vline(x=spot, line_width=2, line_dash="dot", line_color="black",
        annotation_text=f"Spot: {spot}", annotation_position="top left")
    fig.update_layout(barmode='stack',
        title=f"Pain Profile: {sym}  |  Spot: {spot}  |  Max Pain: {max_pain}  |  Distance: {row['Distance%']}%",
        xaxis_title="Strike Price", yaxis_title="Total Pain (₹ × OI)", height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
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
            return go.Figure(), html.Div(), dash.no_update
        return go.Figure(), html.Div(), dash.no_update
    df_heat = _get_heatmap_data()
    if df_heat.empty:
        return go.Figure(), html.Div("No data available."), dash.no_update
    needed_cols = ['Symbol', 'Expiry_Rank', 'ATM_IV', 'Expiry']
    if not all(c in df_heat.columns for c in needed_cols):
        return go.Figure(), html.Div("Data missing required columns (Expiry_Rank, ATM_IV). Try clearing cache."), dash.no_update
    df_atm = df_heat.groupby(['Symbol', 'Expiry_Rank']).agg({
        'ATM_IV': 'first', 'Expiry': 'first', 'Spot': 'first'
    }).reset_index()
    df_pivot = df_atm.pivot(index='Symbol', columns='Expiry_Rank', values=['ATM_IV', 'Expiry', 'Spot'])
    df_pivot.columns = [f'{c[0]}_{c[1]}' for c in df_pivot.columns]
    if 'ATM_IV_1' not in df_pivot.columns or 'ATM_IV_2' not in df_pivot.columns:
         return go.Figure(), html.Div("Insufficient multi-expiry data found."), dash.no_update
    df_spread = df_pivot.dropna(subset=['ATM_IV_1', 'ATM_IV_2']).copy()
    if df_spread.empty:
        return go.Figure(), html.Div("No symbols found with both Near and Far expiry data."), dash.no_update
    df_spread['Spread'] = df_spread['ATM_IV_1'] - df_spread['ATM_IV_2']
    df_spread['Abs_Spread'] = df_spread['Spread'].abs()
    df_spread = df_spread.sort_values('Abs_Spread', ascending=False).reset_index()
    table_df = df_spread[['Symbol', 'ATM_IV_1', 'ATM_IV_2', 'Spread', 'Expiry_1', 'Expiry_2']].copy()
    table_df.columns = ['Symbol', 'Near IV%', 'Far IV%', 'Spread', 'Near Exp', 'Far Exp']
    table_df['Near IV%'] = table_df['Near IV%'].round(2)
    table_df['Far IV%'] = table_df['Far IV%'].round(2)
    table_df['Spread'] = table_df['Spread'].round(2)
    table = dash_table.DataTable(
        id='calendar-table',
        columns=[{'name': c, 'id': c} for c in table_df.columns],
        data=table_df.to_dict('records'),
        sort_action='native', filter_action='native', page_size=15, row_selectable='single',
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'padding': '8px'},
        style_header={'backgroundColor': '#fd7e14', 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'filter_query': '{Spread} > 0', 'column_id': 'Spread'}, 'color': '#dc3545', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{Spread} < 0', 'column_id': 'Spread'}, 'color': '#28a745', 'fontWeight': 'bold'},
        ])
    top_n = df_spread.head(30)
    colors = ['#dc3545' if s > 0 else '#28a745' for s in top_n['Spread']]
    fig = go.Figure(go.Bar(x=top_n['Spread'], y=top_n['Symbol'], orientation='h',
        marker_color=colors, text=top_n['Spread'].round(2), textposition='auto',
        hovertemplate="<b>%{y}</b><br>Spread: %{x:.2f}%<br>Near IV: %{customdata[0]:.2f}%<br>Far IV: %{customdata[1]:.2f}%<extra></extra>",
        customdata=top_n[['ATM_IV_1', 'ATM_IV_2']]))
    fig.update_layout(title=f"Calendar Spread (Near IV - Far IV) — Top {len(top_n)} Divergences",
        xaxis_title="Spread (IV%)", yaxis={'autorange': 'reversed'},
        height=max(600, len(top_n) * 20), template="plotly_white")
    fig.add_vline(x=0, line_width=1, line_color="black")
    relevant_symbols = df_spread['Symbol'].tolist()
    df_store = df_heat[df_heat['Symbol'].isin(relevant_symbols)].copy()
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
    df = pd.DataFrame(stored_data)
    df_sym = df[df['Symbol'] == sym].copy()
    if df_sym.empty:
        return go.Figure(), f"No data found for {sym}"
    df1 = df_sym[df_sym['Expiry_Rank'] == 1].sort_values('Strike')
    df2 = df_sym[df_sym['Expiry_Rank'] == 2].sort_values('Strike')
    exp1 = df1['Expiry'].iloc[0] if not df1.empty else "Near"
    exp2 = df2['Expiry'].iloc[0] if not df2.empty else "Far"
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df1['Strike'], y=df1['IV'], name=f"Near: {exp1}", marker_color='#007bff'))
    fig.add_trace(go.Bar(x=df2['Strike'], y=df2['IV'], name=f"Far: {exp2}", marker_color='#fd7e14'))
    fig.update_layout(title=f"Calendar IV Structure: {sym} ({exp1} vs {exp2})",
        xaxis_title="Strike Price", yaxis_title="Implied Volatility (%)",
        barmode='group', hovermode="x unified", template="plotly_white")
    return fig, f"Strike-wise IV Comparison: {sym}"

# --- LIVE IV ANALYZER CALLBACKS (Preserved) ---

# --- Populate Strike Dropdown for Monitoring ---
@app.callback(
    Output('history-strike-dropdown', 'options'),
    [Input('scrip-input', 'value'), Input('expiry-dropdown', 'value')]
)
def update_monitor_dropdown_options(scrip, expiry):
    if not scrip:
        return []
    try:
        # build_option_chain with enrich=False is instant — no API call,
        # it just filters the already-loaded instruments DataFrame
        meta, chain = build_option_chain(scrip.upper().strip(), expiry=expiry, enrich=False)
        if chain.empty:
            return []
        strikes = sorted(chain['strike'].unique())
        return [{'label': str(s), 'value': s} for s in strikes]
    except Exception:
        return []

# --- LIVE LOGGING AND PLOTTING ---
@app.callback(
    [Output('iv-history-graph', 'figure'),
     Output('monitor-status', 'children'),
     Output('history-strike-dropdown', 'options', allow_duplicate=True)],
    [Input('monitor-interval', 'n_intervals'),
     Input('history-strike-dropdown', 'value')],
    [State('scrip-input', 'value'),
     State('expiry-dropdown', 'value'),
     State('history-strike-dropdown', 'options')],
    prevent_initial_call=True
)
def live_iv_logger_and_plot(n_intervals, selected_strike, scrip, expiry, existing_strike_opts):
    if not scrip or not expiry:
        return go.Figure(), "Select Scrip & Expiry first", existing_strike_opts

    # --- Market hours gate ---
    scrip_exchange = base_name_meta_lookup.get(scrip.upper().strip(), {}).get('exchange', 'NFO')
    mkt_open, mkt_reason = is_market_open(scrip_exchange)
    if not mkt_open:
        fig = go.Figure()
        fig.update_layout(title=f"Market Closed — {mkt_reason}")
        return fig, f"⏸ {mkt_reason}", existing_strike_opts
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    current_time = datetime.now().strftime('%H:%M:%S')
    if trigger_id == 'monitor-interval':
        try:
            df_chain, spot, atm = get_option_data(scrip, expiry=expiry)
            if not df_chain.empty:
                log_df = df_chain[['STRIKE', 'CE-IV', 'PE-IV']].copy()
                log_df['Timestamp'] = current_time
                log_df['Symbol'] = scrip
                header_mode = not os.path.exists(IV_LOG_FILE)
                log_df.to_csv(IV_LOG_FILE, mode='a', header=header_mode, index=False)
                strikes = sorted(log_df['STRIKE'].unique())
                strike_opts = [{'label': str(s), 'value': s} for s in strikes]
            else:
                strike_opts = existing_strike_opts
        except Exception as e:
            return go.Figure(), f"Error fetching: {e}", existing_strike_opts
    else:
        strike_opts = existing_strike_opts
    if not selected_strike:
        fig = go.Figure()
        fig.update_layout(title="Select a Strike to see Live History")
        return fig, f"Last Logged: {current_time}", strike_opts
    if not os.path.exists(IV_LOG_FILE):
        return go.Figure(), "No history log found yet.", strike_opts
    try:
        full_log = pd.read_csv(IV_LOG_FILE)
        mask = (full_log['Symbol'] == scrip) & (full_log['STRIKE'] == selected_strike)
        df_plot = full_log[mask].copy()
        if df_plot.empty:
            return go.Figure(), f"No history for {selected_strike}", strike_opts
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot['Timestamp'], y=df_plot['CE-IV'], mode='lines+markers', name='Call IV', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df_plot['Timestamp'], y=df_plot['PE-IV'], mode='lines+markers', name='Put IV', line=dict(color='red')))
        fig.update_layout(
            title=f"Live IV History: {scrip} {selected_strike} Strike",
            xaxis_title="Time", yaxis_title="IV (%)",
            template="plotly_white", hovermode="x unified")
        return fig, f"Live Updates Active | Last: {current_time}", strike_opts
    except Exception as e:
        return go.Figure(), f"Plot Error: {e}", strike_opts

# --- IV PRO TRADER CALLBACKS ---

@app.callback(
    [Output('ivpro-dd-expiry', 'options'),
     Output('ivpro-dd-expiry', 'value')],
    Input('ivpro-dd-symbol', 'value')
)
def update_ivpro_expiry_options(symbol):
    if not symbol or symbol not in base_name_meta_lookup:
        return [], None
    # base_name_meta_lookup uses 'options_expiries'
    expiries = base_name_meta_lookup[symbol]['options_expiries']
    # Format: YYYY-MON-DD
    opts = [{'label': e.strftime('%Y-%b-%d').upper(), 'value': str(e.date())} for e in expiries]
    default = opts[0]['value'] if opts else None
    return opts, default

@app.callback(
    Output('ivpro-dd-strike', 'options'),
    [Input('ivpro-dd-symbol', 'value'),
     Input('ivpro-dd-expiry', 'value')]
)
def update_ivpro_strike_options(symbol, expiry):
    conn = sqlite3.connect(IV_DB_FILE)
    try:
        if expiry:
            query = "SELECT DISTINCT strike FROM iv_history WHERE symbol = ? AND expiry = ? ORDER BY strike"
            df = pd.read_sql_query(query, conn, params=(symbol, expiry))
        else:
            query = "SELECT DISTINCT strike FROM iv_history WHERE symbol = ? ORDER BY strike"
            df = pd.read_sql_query(query, conn, params=(symbol,))
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    
    if df.empty: return []
    return [{'label': str(s), 'value': s} for s in df['strike']]

@app.callback(
    Output('ivpro-store-tf', 'data'),
    [Input('ivpro-btn-1d', 'n_clicks'), Input('ivpro-btn-5d', 'n_clicks'), 
     Input('ivpro-btn-1m', 'n_clicks'), Input('ivpro-btn-all', 'n_clicks')]
)
def set_ivpro_timeframe(b1, b5, b30, ball):
    ctx = callback_context
    if not ctx.triggered: return '1D'
    btn_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if btn_id == 'ivpro-btn-1d': return '1D'
    if btn_id == 'ivpro-btn-5d': return '5D'
    if btn_id == 'ivpro-btn-1m': return '1M'
    return 'ALL'

@app.callback(
    Output('ivpro-clock', 'children'),
    [Input('ivpro-interval', 'n_intervals')]
)
def update_ivpro_clock(n):
    return datetime.now().strftime("%d %b %H:%M")

@app.callback(
    [Output('ivpro-ce-chart', 'figure'),
     Output('ivpro-pe-chart', 'figure')],
    [Input('ivpro-interval', 'n_intervals'),
     Input('ivpro-dd-symbol', 'value'),
     Input('ivpro-dd-strike', 'value'),
     Input('ivpro-dd-expiry', 'value'),
     Input('ivpro-store-tf', 'data')]
)
def update_ivpro_charts(n, symbol, strike, expiry, timeframe):
    dark_layout = dict(
        template='plotly_dark',
        paper_bgcolor='#0a0a14', plot_bgcolor='#0a0a14',
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)'),
        hovermode='x unified',
        margin=dict(l=50, r=20, t=40, b=30),
        legend=dict(orientation='h', y=1, x=0, bgcolor='rgba(0,0,0,0)')
    )
    
    empty_fig = go.Figure()
    empty_fig.update_layout(**dark_layout)
    
    if not symbol or not strike:
        empty_fig.update_layout(title='Select a Symbol, Expiry, and Strike to view History')
        return empty_fig, empty_fig
    
    # Timeframe Logic
    limit_clause = ""
    params = [symbol, float(strike)]
    
    if expiry:
        limit_clause += " AND expiry = ?"
        params.append(expiry)
    
    if timeframe == '1D':
        start_dt = datetime.now().strftime('%Y-%m-%d 00:00:00')
        limit_clause += " AND timestamp >= ?"
        params.append(start_dt)
    elif timeframe == '5D':
        start_dt = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d 00:00:00')
        limit_clause += " AND timestamp >= ?"
        params.append(start_dt)
    elif timeframe == '1M':
        start_dt = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d 00:00:00')
        limit_clause += " AND timestamp >= ?"
        params.append(start_dt)
    
    conn = sqlite3.connect(IV_DB_FILE)
    try:
        query = f"""
            SELECT timestamp, ce_iv, pe_iv 
            FROM iv_history 
            WHERE symbol = ? AND strike = ? {limit_clause}
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn, params=params)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    
    if df.empty:
        empty_fig.update_layout(title=f'No data yet for {symbol} {strike} ({timeframe})')
        return empty_fig, empty_fig
    
    exp_label = f" | Exp: {expiry}" if expiry else ""
    
    # --- Call IV Chart ---
    ce_valid = df['ce_iv'].replace(0, np.nan)
    ce_fig = go.Figure()
    ce_fig.add_trace(go.Scatter(
        x=df['timestamp'], y=ce_valid,
        mode='lines', name='Call IV',
        line=dict(color='#00e676', width=2),
        connectgaps=True
    ))
    ce_min = ce_valid.min() if ce_valid.notna().any() else 0
    ce_max = ce_valid.max() if ce_valid.notna().any() else 100
    ce_pad = max((ce_max - ce_min) * 0.15, 1)
    ce_fig.update_layout(
        title=f"Call IV: {symbol} {strike} Strike ({timeframe}){exp_label}",
        **dark_layout,
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title='IV %',
                   range=[max(0, ce_min - ce_pad), ce_max + ce_pad])
    )
    
    # --- Put IV Chart ---
    pe_valid = df['pe_iv'].replace(0, np.nan)
    pe_fig = go.Figure()
    pe_fig.add_trace(go.Scatter(
        x=df['timestamp'], y=pe_valid,
        mode='lines', name='Put IV',
        line=dict(color='#ff1744', width=2),
        connectgaps=True
    ))
    pe_min = pe_valid.min() if pe_valid.notna().any() else 0
    pe_max = pe_valid.max() if pe_valid.notna().any() else 100
    pe_pad = max((pe_max - pe_min) * 0.15, 1)
    pe_fig.update_layout(
        title=f"Put IV: {symbol} {strike} Strike ({timeframe}){exp_label}",
        **dark_layout,
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title='IV %',
                   range=[max(0, pe_min - pe_pad), pe_max + pe_pad])
    )
    
    return ce_fig, pe_fig

# ==============================================================================
# 6. SERVER RUN
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)




