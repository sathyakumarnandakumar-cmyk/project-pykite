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

# Global variable to cache instruments in memory for fast searching
INSTRUMENTS_CACHE = []

@app.on_event("startup")
async def startup_event():
    """Fetches the instrument list once when the server starts to optimize performance."""
    global INSTRUMENTS_CACHE
    try:
        # Initializing the cache with the full instrument list
        INSTRUMENTS_CACHE = kite.instruments()
        print(f"Successfully cached {len(INSTRUMENTS_CACHE)} instruments.")
    except Exception as e:
        print(f"Error caching instruments: {e}")

# --- AUTHENTICATION UTILS ---

@app.get("/auth/login_url")
def get_login_url():
    """Generates the login URL to get the request_token."""
    return {"login_url": kite.login_url()}

@app.get("/auth/generate_session")
def generate_session(request_token: str):
    """Exchanges request_token for a permanent access_token for the day."""
    try:
        # Note: API_SECRET should be loaded from your environment variables
        api_secret = os.getenv("API_SECRET")
        data = kite.generate_session(request_token, api_secret=api_secret)
        kite.set_access_token(data["access_token"])
        return {"status": "success", "user_data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- MARKET DATA & SEARCH (OPTIMIZED) ---

@app.get("/market/instruments")
def get_instruments(exchange: Optional[str] = None):
    """Returns instruments from the local cache instead of fetching from Zerodha."""
    if exchange:
        filtered = [i for i in INSTRUMENTS_CACHE if i.get('exchange') == exchange.upper()]
        return {"count": len(filtered), "instruments": filtered}
    return {"count": len(INSTRUMENTS_CACHE), "instruments": INSTRUMENTS_CACHE}

@app.get("/search/instruments")
def search_instruments(query: str):
    """Searches the local cache by symbol or name for near-instant results."""
    try:
        query = query.lower()
        results = [
            i for i in INSTRUMENTS_CACHE 
            if query in i.get('tradingsymbol', '').lower() or query in i.get('name', '').lower()
        ]
        return {"count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- ORDER MANAGEMENT (FIXED) ---

@app.get("/orders/{order_id}")
def get_order_details(order_id: str):
    """Fetch specific order history directly using order_id for efficiency."""
    try:
        # Optimized from full list search to direct history lookup
        return kite.order_history(order_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- PORTFOLIO & TRADING (RETAINED) ---

@app.get("/user/margins")
def get_margins(segment: Optional[str] = None):
    return kite.margins(segment) if segment else kite.margins()

@app.get("/portfolio/summary")
def get_portfolio_summary():
    try:
        return {
            "holdings": kite.holdings(),
            "positions": kite.positions()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/orders/place")
def place_order(
    symbol: str, exchange: str, transaction_type: str, quantity: int, 
    product: str = "MIS", order_type: str = "MARKET"
):
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