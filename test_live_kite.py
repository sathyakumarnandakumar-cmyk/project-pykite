"""
Test 3: Live Kite endpoints — positions, profile, health after login.
Run while server is up + logged in:  python test_live_kite.py
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
    data = resp.json()

    # Pretty print for nested data
    if isinstance(data, dict) and any(isinstance(v, (list, dict)) for v in data.values()):
        import json
        print(f"Body:\n{json.dumps(data, indent=2, default=str)[:2000]}")
    else:
        print(f"Body:   {data}")
    return resp


# ── 1. Confirm session is live ──
print("🔷 Checking Kite session status...")
r = test("get", "/kite/app")
if not r.json().get("kite_connected"):
    print("\n⚠️  Kite not connected. Run test_health_login.py first to login.")
    exit(1)

# ── 2. Full health check ──
print("\n🔷 Health check...")
test("get", "/health")

# ── 3. Profile ──
print("\n🔷 Fetching profile...")
test("get", "/profile")

# ── 4. Positions ──
print("\n🔷 Fetching live positions...")
r = test("get", "/positions")
net = r.json().get("net", [])
print(f"\n   📊 {len(net)} net positions found")
if net:
    for p in net[:5]:
        pnl = p.get("pnl", 0)
        sym = p.get("tradingsymbol", "?")
        qty = p.get("quantity", 0)
        print(f"     {sym}: qty={qty}  pnl={pnl}")

print("\n✅ All live Kite tests complete!")
