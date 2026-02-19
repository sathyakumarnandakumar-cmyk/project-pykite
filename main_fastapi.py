from fastapi import FastAPI, HTTPException
import sys
import os
import importlib
from contextlib import asynccontextmanager

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import local modules
import login_test
import load_kite_from_access

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the kite session on startup
    print("Application startup: Loading Kite session...")
    yield
    print("Application shutdown")

app = FastAPI(title="Kite Connect Cloud Login", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Kite Connect Cloud Login Service is running"}

@app.get("/login")
def trigger_login():
    """
    Triggers the login flow. 
    Note: On Cloud Run, this will print the URL to logs. 
    You might need to copy-paste the URL manually if not running locally.
    """
    try:
        login_test.login_kite()
        return {"status": "success", "message": "Login flow triggered. Check logs for URL if not opened automatically."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profile")
def get_profile():
    """
    Returns the user profile if logged in.
    """
    try:
        # Reload to ensure we have the latest token if login just happened
        importlib.reload(load_kite_from_access)
        kite = load_kite_from_access.kite
        
        if not kite:
             return {"status": "error", "message": "Kite session not established. Call /login first."}
             
        profile = kite.profile()
        return {"status": "success", "profile": profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
def run_full_test():
    """
    Runs the full verification logic from login_test.py
    """
    try:
        profile = login_test.get_kite_profile()
        if profile:
            return {"status": "success", "data": profile}
        else:
             return {"status": "error", "message": "Failed to get profile. Try /login first."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
