"""
api_backend.py — FastAPI backend for strategy group management.

Run:
    uvicorn api_backend:app --reload --port 8000

Baby-step scope: CRUD for groups and legs, order log viewing.
Stop-loss / trailing triggers will be added later.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import asynccontextmanager
import sys, os, sqlite3

# ── Path setup ──
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import strategy_group as sg

import load_kite_from_access
_KITE_AVAILABLE = True


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "trading_data.db")
APP_NAME = os.getenv("KITE_APP_NAME", "algotestlive")


# ─────────────────────────────────────────────
#  Lifespan — DB + Kite session
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.db = sg.init_db(DB_PATH)
    app.state.app_name = APP_NAME
    app.state.kite = None
    app.state.kite_profile = None
    print(f"📂 Database: {os.path.abspath(DB_PATH)}")

    if _KITE_AVAILABLE:
        _load_kite_session(app, APP_NAME)
    else:
        print("⚠️ kiteconnect not installed — Kite endpoints disabled")

    yield

    # Shutdown
    app.state.db.close()
    print("🛑 Shutdown — DB closed.")


def _load_kite_session(app_instance, app_name: str):
    """
    Load (or reload) a Kite session for the given app_name.
    Stores kite instance, profile, and app_name in app.state.
    """
    if not _KITE_AVAILABLE:
        raise HTTPException(503, "kiteconnect is not installed.")

    kite = load_kite_from_access.get_kite_session(app_name)
    app_instance.state.kite = kite
    app_instance.state.app_name = app_name

    if kite:
        try:
            profile = kite.profile()
            app_instance.state.kite_profile = profile
            print(f"✅ Kite session active: {profile['user_name']} "
                  f"(ID: {profile['user_id']}) via {app_name}")
        except Exception:
            app_instance.state.kite_profile = None
            print(f"⚠️ Kite session loaded for {app_name} but profile fetch failed")
    else:
        app_instance.state.kite_profile = None
        print(f"⚠️ Kite session unavailable for {app_name}")

    return kite


# ─────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────

app = FastAPI(
    title="AlgoArms Strategy API",
    description="Manage multi-leg strategy groups, view positions and order logs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Pydantic Models
# ─────────────────────────────────────────────

class LegIn(BaseModel):
    symbol: str = Field(..., example="GOLDM26APRFUT")
    exchange: str = Field(..., example="MCX")
    quantity: int = Field(..., example=1)
    transaction_type: str = Field(..., example="BUY",
                                  description="BUY or SELL")
    entry_price: float = Field(..., example=142177.0)
    product: str = Field("NRML", example="NRML")


class GroupCreateIn(BaseModel):
    name: str = Field(..., example="GOLDM Synthetic Put")
    trigger_mode: str = Field(..., example="underlying",
                              description="'combined_pnl' or 'underlying'")
    trailing_percent: float = Field(..., example=3.0)
    ref_symbol: Optional[str] = Field(None, example="GOLDM26APRFUT")
    ref_exchange: Optional[str] = Field(None, example="MCX")
    legs: Optional[List[LegIn]] = None


class LegAddIn(BaseModel):
    symbol: str
    exchange: str
    quantity: int
    transaction_type: str
    entry_price: float
    product: str = "NRML"


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _db():
    return app.state.db


def _kite():
    k = app.state.kite
    if not k:
        raise HTTPException(503, "Kite session not available. "
                            "Login first or check APP_NAME.")
    return k


def _row_to_dict(row):
    """Convert sqlite3.Row to a plain dict."""
    return dict(row) if row else None


# ─────────────────────────────────────────────
#  Routes — Health
# ─────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "AlgoArms Strategy API v0.1"}


@app.get("/health", tags=["Health"])
def health():
    """Full health check — Kite session, DB, active groups count."""
    kite_ok = app.state.kite is not None
    profile = app.state.kite_profile

    # Count active groups
    cur = _db().cursor()
    cur.execute("SELECT COUNT(*) as cnt FROM strategy_groups WHERE status = 'active'")
    active_groups = cur.fetchone()["cnt"]

    return {
        "status": "ok",
        "kite_connected": kite_ok,
        "app_name": app.state.app_name,
        "user_name": profile["user_name"] if profile else None,
        "user_id": profile["user_id"] if profile else None,
        "db_path": os.path.abspath(DB_PATH),
        "active_groups": active_groups,
    }


# ─────────────────────────────────────────────
#  Routes — Kite Session Management
# ─────────────────────────────────────────────

@app.get("/kite/app", tags=["Kite Session"])
def get_kite_app():
    """
    Show which Kite app is currently loaded, session status, and user info.
    """
    profile = app.state.kite_profile
    return {
        "app_name": app.state.app_name,
        "kite_connected": app.state.kite is not None,
        "user_name": profile["user_name"] if profile else None,
        "user_id": profile["user_id"] if profile else None,
        "email": profile.get("email") if profile else None,
        "broker": profile.get("broker") if profile else None,
        "exchanges": profile.get("exchanges") if profile else None,
        "products": profile.get("products") if profile else None,
    }


@app.post("/kite/login-access", tags=["Kite Session"])
def kite_login_access(app_name: str = Query(..., description="App name from secrets.ini, e.g. 'algotestlive'")):
    """
    Load (or switch) the Kite session using an EXISTING access token from access.ini.
    This does NOT open a browser — it just picks up a previously saved token.
    If the token is expired, use POST /kite/login to do a fresh OAuth login.
    """
    try:
        kite = _load_kite_session(app, app_name)
        if kite:
            profile = app.state.kite_profile or {}
            return {
                "status": "connected",
                "app_name": app_name,
                "user_name": profile.get("user_name"),
                "user_id": profile.get("user_id"),
            }
        else:
            return {
                "status": "failed",
                "app_name": app_name,
                "message": f"Could not establish session for '{app_name}'. "
                           f"Token may be expired — use POST /kite/login to do a fresh login."
            }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/kite/login", tags=["Kite Session"])
def kite_login(app_name: str = Query(..., description="App name from secrets.ini, e.g. 'algotestlive'")):
    """
    Full OAuth login: opens the Kite login page in the server's browser,
    starts a local callback server to capture the request_token,
    generates an access_token, saves it, and loads the session.

    Flow:
    1. Opens https://kite.zerodha.com/connect/login in the browser
    2. You login on that page → Kite redirects to localhost callback
    3. Callback captures request_token → generates access_token
    4. Token saved to access.ini → session loaded into the API
    """
    if not _KITE_AVAILABLE:
        raise HTTPException(503, "kiteconnect is not installed.")

    import threading

    try:
        import get_token_local
    except ImportError:
        raise HTTPException(503, "get_token_local module not found.")

    def _run_login():
        """Run the blocking login flow in a background thread."""
        try:
            get_token_local.login(app_name)
            # After login completes, auto-load the session
            _load_kite_session(app, app_name)
            print(f"✅ Auto-loaded session after login for {app_name}")
        except Exception as e:
            print(f"❌ Login thread error: {e}")

    # Start login in background (it blocks waiting for browser redirect)
    thread = threading.Thread(target=_run_login, daemon=True)
    thread.start()

    # Fetch the login URL so we can return it
    try:
        from config import get_app_config
        app_conf = get_app_config(app_name)
        api_key = app_conf["KITE_API_KEY"]
        login_url = f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"
    except Exception:
        login_url = None

    return {
        "status": "login_initiated",
        "app_name": app_name,
        "message": "Browser login page opened. Complete the login in your browser. "
                   "Once done, the session will be auto-loaded.",
        "login_url": login_url,
    }


# ─────────────────────────────────────────────
#  Routes — Groups
# ─────────────────────────────────────────────

@app.post("/groups", tags=["Groups"], status_code=201)
def create_group(body: GroupCreateIn):
    """Create a new strategy group, optionally with legs."""
    try:
        legs_dicts = [l.model_dump() for l in body.legs] if body.legs else None
        group_id = sg.create_group(
            _db(),
            name=body.name,
            trigger_mode=body.trigger_mode,
            trailing_percent=body.trailing_percent,
            ref_symbol=body.ref_symbol,
            ref_exchange=body.ref_exchange,
            legs=legs_dicts,
        )
        return {"status": "created", "group_id": group_id}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/groups", tags=["Groups"])
def list_groups(status: str = Query("active",
                                    description="Filter: active | closed | all")):
    """List strategy groups."""
    cur = _db().cursor()
    if status == "all":
        cur.execute("SELECT * FROM strategy_groups ORDER BY id DESC")
    else:
        cur.execute("SELECT * FROM strategy_groups WHERE status = ? ORDER BY id DESC",
                    (status,))
    rows = [_row_to_dict(r) for r in cur.fetchall()]
    return {"count": len(rows), "groups": rows}


@app.get("/groups/{group_id}", tags=["Groups"])
def get_group(group_id: int):
    """Get a single group with its legs."""
    cur = _db().cursor()

    cur.execute("SELECT * FROM strategy_groups WHERE id = ?", (group_id,))
    group = _row_to_dict(cur.fetchone())
    if not group:
        raise HTTPException(404, f"Group {group_id} not found.")

    cur.execute("""
        SELECT * FROM strategy_group_legs
        WHERE group_id = ? ORDER BY id
    """, (group_id,))
    legs = [_row_to_dict(r) for r in cur.fetchall()]

    return {**group, "legs": legs}


@app.delete("/groups/{group_id}", tags=["Groups"])
def close_group(group_id: int):
    """Soft-close a group and all its active legs."""
    cur = _db().cursor()

    cur.execute("SELECT id FROM strategy_groups WHERE id = ?", (group_id,))
    if not cur.fetchone():
        raise HTTPException(404, f"Group {group_id} not found.")

    cur.execute("""
        UPDATE strategy_groups
        SET status = 'closed', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (group_id,))
    cur.execute("""
        UPDATE strategy_group_legs
        SET status = 'closed'
        WHERE group_id = ? AND status = 'active'
    """, (group_id,))
    _db().commit()

    return {"status": "closed", "group_id": group_id}


# ─────────────────────────────────────────────
#  Routes — Legs
# ─────────────────────────────────────────────

@app.post("/groups/{group_id}/legs", tags=["Legs"], status_code=201)
def add_leg_to_group(group_id: int, body: LegAddIn):
    """Add a leg to an existing group. Removes matching individual TSL."""
    cur = _db().cursor()
    cur.execute("SELECT id FROM strategy_groups WHERE id = ? AND status = 'active'",
                (group_id,))
    if not cur.fetchone():
        raise HTTPException(404, f"Active group {group_id} not found.")

    try:
        leg_id = sg.add_leg(
            _db(), group_id,
            symbol=body.symbol,
            exchange=body.exchange,
            quantity=body.quantity,
            transaction_type=body.transaction_type,
            entry_price=body.entry_price,
            product=body.product,
        )
        return {"status": "added", "leg_id": leg_id, "group_id": group_id}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/legs/{leg_id}", tags=["Legs"])
def remove_leg_endpoint(leg_id: int):
    """Soft-close a single leg."""
    sg.remove_leg(_db(), leg_id)
    return {"status": "closed", "leg_id": leg_id}


@app.get("/groups/{group_id}/legs", tags=["Legs"])
def list_legs(group_id: int,
              status: str = Query("active",
                                  description="Filter: active | closed | all")):
    """List legs for a group."""
    cur = _db().cursor()
    if status == "all":
        cur.execute("""
            SELECT * FROM strategy_group_legs
            WHERE group_id = ? ORDER BY id
        """, (group_id,))
    else:
        cur.execute("""
            SELECT * FROM strategy_group_legs
            WHERE group_id = ? AND status = ? ORDER BY id
        """, (group_id, status))
    rows = [_row_to_dict(r) for r in cur.fetchall()]
    return {"count": len(rows), "legs": rows}


# ─────────────────────────────────────────────
#  Routes — Order Log
# ─────────────────────────────────────────────

@app.get("/orders", tags=["Orders"])
def get_orders(limit: int = Query(50, ge=1, le=500),
               source: Optional[str] = Query(None,
                   description="Filter by source: individual | group_exit | tsl_gtt | manual")):
    """View the order log (all orders placed through the system)."""
    rows = sg.get_order_log(_db(), limit=limit, source=source)
    return {"count": len(rows), "orders": rows}


# ─────────────────────────────────────────────
#  Routes — Positions (live from Kite)
# ─────────────────────────────────────────────

@app.get("/positions", tags=["Kite Live"])
def get_positions():
    """Fetch live positions from Kite."""
    try:
        positions = _kite().positions()
        net = positions.get("net", [])
        day = positions.get("day", [])
        return {"net_count": len(net), "day_count": len(day),
                "net": net, "day": day}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/profile", tags=["Kite Live"])
def get_profile():
    """Fetch user profile from Kite."""
    try:
        profile = _kite().profile()
        return {"status": "ok", "profile": profile}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_backend:app", host="0.0.0.0", port=8000, reload=True)
