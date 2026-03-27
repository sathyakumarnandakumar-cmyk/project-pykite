import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sqlite3
import os

# Import the app
from api_backend import app, DB_PATH
import strategy_group as sg

# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def mem_db():
    """Create an in-memory SQLite database for testing."""
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    
    # Initialize strategy_group tables
    db.executescript(sg._SCHEMA_SQL)
    
    # Initialize the missing table that `add_leg` updates
    db.executescript("""
        CREATE TABLE IF NOT EXISTS active_tsl_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            exchange TEXT,
            status TEXT,
            updated_at DATETIME
        );
    """)
    db.commit()
    
    yield db
    db.close()

@pytest.fixture(scope="module")
def client(mem_db):
    """
    Provide a FastAPI TestClient with mocked Database and Kite Session.
    We patch the initialization functions to return our test mocks.
    """
    mock_kite = MagicMock()
    # Mocking Kite endpoint returns
    mock_kite.positions.return_value = {
        "net": [{"tradingsymbol": "MOCK_NET", "quantity": 1}],
        "day": [{"tradingsymbol": "MOCK_DAY", "quantity": 2}]
    }
    mock_kite.profile.return_value = {"user_name": "Test User", "email": "test@example.com"}

    # We patch `api_backend` imports to ensure lifespan picks up our mocks
    with patch("api_backend.sg.init_db", return_value=mem_db):
        with patch("api_backend.load_kite_from_access.get_kite_session", return_value=mock_kite, create=True):
            with patch("api_backend._KITE_AVAILABLE", True):
                with TestClient(app) as test_client:
                    yield test_client

# ─────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["kite_connected"] is True


def test_create_group(client):
    payload = {
        "name": "Test Synthetic Put",
        "trigger_mode": "combined_pnl",
        "trailing_percent": 3.0
    }
    response = client.post("/groups", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "created"
    assert "group_id" in data
    assert isinstance(data["group_id"], int)


def test_list_groups(client):
    response = client.get("/groups?status=all")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    assert any(g["name"] == "Test Synthetic Put" for g in data["groups"])


def test_add_leg(client):
    # Retrieve the first group from the DB to add a leg to
    groups_resp = client.get("/groups?status=active")
    group_id = groups_resp.json()["groups"][0]["id"]

    leg_payload = {
        "symbol": "BANKNIFTY",
        "exchange": "NFO",
        "quantity": 15,
        "transaction_type": "BUY",
        "entry_price": 45000.0,
        "product": "NRML"
    }

    response = client.post(f"/groups/{group_id}/legs", json=leg_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "added"
    assert "leg_id" in data


def test_get_group_with_legs(client):
    # Fetch the group and confirm the leg exists
    groups_resp = client.get("/groups?status=active")
    group_id = groups_resp.json()["groups"][0]["id"]
    
    response = client.get(f"/groups/{group_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == group_id
    assert len(data["legs"]) == 1
    assert data["legs"][0]["symbol"] == "BANKNIFTY"
    assert data["legs"][0]["transaction_type"] == "BUY"


def test_positions(client):
    # Since we mocked the kite session, this should return our mock payload
    response = client.get("/positions")
    assert response.status_code == 200
    data = response.json()
    assert data["net_count"] == 1
    assert data["day_count"] == 1
    assert data["net"][0]["tradingsymbol"] == "MOCK_NET"


def test_close_group(client):
    groups_resp = client.get("/groups?status=active")
    group_id = groups_resp.json()["groups"][0]["id"]

    response = client.delete(f"/groups/{group_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "closed"

    # Verify it is closed
    group_verify = client.get(f"/groups/{group_id}")
    assert group_verify.json()["status"] == "closed"
