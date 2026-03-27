"""
Quick smoke test — health + kite session endpoints only.
Run while server is up:  python test_health_login.py
"""

import requests

BASE = "http://localhost:8000"

def test(method, path, **kwargs):
    url = f"{BASE}{path}"
    print(f"\n{'='*50}")
    print(f"{method.upper()} {path}")
    print(f"{'='*50}")
    resp = getattr(requests, method)(url, **kwargs)
    print(f"Status: {resp.status_code}")
    print(f"Body:   {resp.json()}")
    return resp


# 1. Root
test("get", "/")

# 2. Health
test("get", "/health")

# 3. Which Kite app?
test("get", "/kite/app")

# 4. Load session from saved access token
test("post", "/kite/login-access", params={"app_name": "algotestlive"})

# 5. Full OAuth login (opens browser — login there, session auto-loads)
test("post", "/kite/login", params={"app_name": "algotestlive"})
