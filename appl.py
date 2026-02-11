# ==============================================================================
# 1. IMPORTS & SETUP
# ==============================================================================
import sys
import pandas as pd
import numpy as np
from datetime import datetime, date
import warnings
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go

# Volatility Imports
from py_vollib.black_scholes.implied_volatility import implied_volatility as bsm_iv

# --- IMPORT KITE ---
# We assume load_kite_from_access.py is in the same folder
try:
    import load_kite_from_access
    kite = load_kite_from_access.kite
    print(f"✅ Connected to Kite: {kite.profile()['user_name']}")
except Exception as e:
    print(f"❌ Error importing Kite: {e}")
    print("Ensure 'load_kite_from_access.py' is in this folder and login is valid.")
    sys.exit(1) # Stop app if login fails

# ==============================================================================
# 2. GLOBAL DATA LOADING (Runs once when server starts)
# ==============================================================================
print("\n⏳ Initializing: Fetching Instruments & Building Lookups... (This takes a moment)")

try:
    # 1. Fetch Instruments
    instruments_nse = kite.instruments("NSE")
    instruments_nfo = kite.instruments("NFO")
    instruments_mcx = kite.instruments("MCX")
    
    instruments_all = instruments_nse + instruments_nfo + instruments_mcx
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
    
    # Separate MCX for specific logic
    df_mcx = df_all[df_all['exchange'] == 'MCX'].copy()
    
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
    
    print(f"✅ Initialization Complete. Loaded {len(df_all)} instruments.")

except Exception as e:
    print(f"❌ CRITICAL ERROR during data loading: {e}")
    sys.exit(1)

# ==============================================================================
# 3. CORE FUNCTIONS (Copied from your Notebook)
# ==============================================================================

def build_option_chain(symbol, expiry=None, enrich=False):
    # 1. Validate symbol exists
    if symbol not in base_name_meta_lookup:
        print(f"❌ Symbol '{symbol}' not found")
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
    source_df = df_mcx if exchange == 'MCX' else df_all
    
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
    if option_chain.empty: return option_chain
    
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

        else:
            # NSE Spot
            spot_sym = f"NSE:{basename}" if basename != 'NIFTY' and basename != 'BANKNIFTY' else f"NSE:{basename} 50"
            if basename == 'NIFTY': spot_sym = "NSE:NIFTY 50"
            if basename == 'BANKNIFTY': spot_sym = "NSE:NIFTY BANK"
            
            ltp_data = kite.ltp(spot_sym)
            spot_price = ltp_data[spot_sym]['last_price']
    except:
        spot_price = 0
        
    if spot_price == 0: return option_chain # Cannot calc IV without spot
    
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
    return option_chain, spot_price

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

def get_option_data(scrip_name):
    # The Master Function called by Dash
    scrip_name = scrip_name.upper().strip()
    
    # 1. Build & Enrich
    meta, chain = build_option_chain(scrip_name, enrich=True)
    
    if chain.empty: return pd.DataFrame()
    
    # 2. Calculate IV
    chain, spot_price = calculate_iv_vollib(chain, metadata=meta, oi_filter_pct=0.95)
    
    # 3. Format
    df_formatted = show_chain(chain, scrip_name)
    
    return df_formatted, spot_price

# ==============================================================================
# 4. DASH WEB APP CONFIGURATION
# ==============================================================================
app = Dash(__name__)
app.title = "Kite Option Smile"

app.layout = html.Div(style={'fontFamily': 'sans-serif', 'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px'}, children=[
    
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

@app.callback(
    [Output('smile-graph', 'figure'), Output('error-msg', 'children')],
    [Input('submit-btn', 'n_clicks')],
    [State('scrip-input', 'value')]
)
def update_dashboard(n_clicks, scrip_name):
    if not scrip_name:
        return go.Figure(), ""
        
    try:
        # RUN THE LOGIC
        df, spot_price = get_option_data(scrip_name)
        
        if df.empty:
            return go.Figure(), f"No data found for {scrip_name}. Check spelling or market status."
        
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
            fig.add_vline(
                x=spot_price, 
                line_width=2, 
                line_dash="dash", 
                line_color="black",
                annotation_text=f"Spot: {spot_price}", 
                annotation_position="top right"
            )
        
        return fig, ""
        
    except Exception as e:
        return go.Figure(), f"An error occurred: {str(e)}"

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

# ==============================================================================
# 5. SERVER RUN
# ==============================================================================
if __name__ == '__main__':
    # debug=False is safer when using global variables in complex scripts
    app.run(debug=True, use_reloader=False)