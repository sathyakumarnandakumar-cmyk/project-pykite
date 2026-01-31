import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from kiteconnect import KiteConnect
from dotenv import load_dotenv


app = FastAPI(title="Kite Connect Utility API")

# Initialize Kite Client
from load_kite_from_access import kite
if kite is None:
    raise Exception("Failed to initialize Kite session. Check your credentials.")   


# --- AUTHENTICATION UTILS ---

@app.get("/auth/login_url")
def get_login_url():
    """Generates the login URL to get the request_token."""
    return {"login_url": kite.login_url()}


# --- PORTFOLIO & ACCOUNT UTILS ---

@app.get("/user/profile")
def get_profile():
    return kite.profile()

@app.get("/user/margins")
def get_margins(segment: Optional[str] = None):
    return kite.margins(segment) if segment else kite.margins()

@app.get("/portfolio/holdings")
def get_holdings():
    return kite.holdings()

    
# --- MARKET DATA UTILS (ADDITIONAL) ---

@app.get("/market/instruments")
def get_instruments(exchange: Optional[str] = None):
    """
    Fetches list of all tradeable instruments.
    Optional: ?exchange=NSE or ?exchange=BSE or ?exchange=NFO
    """
    try:
        instruments = kite.instruments(exchange) if exchange else kite.instruments()
        return {"count": len(instruments), "instruments": instruments}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/market/ohlc")
def get_ohlc(instruments: List[str] = Query(...)):
    """Get OHLC data. Example: ?instruments=NSE:INFY&instruments=NSE:TCS"""
    try:
        return kite.ohlc(instruments)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- ORDER MANAGEMENT (ADDITIONAL) ---

@app.get("/orders/{order_id}")
def get_order_details(order_id: str):
    """Fetch details of a specific order"""
    try:
        orders = kite.orders()
        return next((o for o in orders if o['order_id'] == order_id), None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/orders/{order_id}/cancel")
def cancel_order(order_id: str):
    """Cancel an existing order"""
    try:
        return kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=order_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/trades")
def get_trades():
    """Fetch all executed trades"""
    return kite.trades()

# --- PORTFOLIO ANALYSIS ---

@app.get("/portfolio/summary")
def get_portfolio_summary():
    """Get overall portfolio statistics"""
    try:
        holdings = kite.holdings()
        positions = kite.positions()
        return {
            "holdings_count": len(holdings),
            "positions_count": len(positions),
            "holdings": holdings,
            "positions": positions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- SEARCH & LOOKUP ---

@app.get("/search/instruments")
def search_instruments(query: str):
    """Search for instruments by symbol or company name"""
    try:
        instruments = kite.instruments()
        results = [i for i in instruments if query.lower() in i.get('tradingsymbol', '').lower()]
        return {"count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
        
@app.get("/portfolio/positions")
def get_positions():
    return kite.positions()

# --- ORDER MANAGEMENT ---

@app.get("/orders")
def get_orders():
    return kite.orders()

@app.post("/orders/place")
def place_order(
    symbol: str, 
    exchange: str, 
    transaction_type: str, 
    quantity: int, 
    product: str = "MIS", 
    order_type: str = "MARKET"
):
    """
    Example: symbol='INFY', exchange='NSE', transaction_type='BUY', quantity=1
    """
    try:
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=exchange,
            tradingsymbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            product=product,
            order_type=order_type
        )
        return {"order_id": order_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- MARKET DATA UTILS ---

@app.get("/market/quote")
def get_quote(instruments: List[str] = Query(...)):
    """Pass multiple instruments like: ?instruments=NSE:INFY&instruments=NSE:RELIANCE"""
    return kite.quote(instruments)

@app.get("/market/ltp")
def get_ltp(instruments: List[str] = Query(...)):
    return kite.ltp(instruments)

@app.get("/market/historical")
def get_historical(
    instrument_token: int, 
    from_date: str, 
    to_date: str, 
    interval: str = "minute"
):
    """Date format: YYYY-MM-DD"""
    try:
        return kite.historical_data(instrument_token, from_date, to_date, interval)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

