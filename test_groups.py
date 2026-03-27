"""
Test 2: Groups & Legs CRUD — create, add legs, list, close.
Run while server is up:  python test_groups.py
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


# ── 1. Create a group with 2 legs (Synthetic Put) ──
print("\n🔷 Creating GOLDM Synthetic Put group with 2 legs...")
r = test("post", "/groups", json={
    "name": "GOLDM Synthetic Put Test",
    "trigger_mode": "underlying",
    "trailing_percent": 3.0,
    "ref_symbol": "GOLDM26APRFUT",
    "ref_exchange": "MCX",
    "legs": [
        {
            "symbol": "GOLDM26APRFUT",
            "exchange": "MCX",
            "quantity": 1,
            "transaction_type": "BUY",
            "entry_price": 142177.0,
            "product": "NRML"
        },
        {
            "symbol": "GOLDM26MAR142000CE",
            "exchange": "MCX",
            "quantity": 1,
            "transaction_type": "SELL",
            "entry_price": 537.5,
            "product": "NRML"
        }
    ]
})
group_id = r.json().get("group_id")
print(f"\n📌 Created group_id = {group_id}")

# ── 2. List active groups ──
print("\n🔷 Listing active groups...")
test("get", "/groups?status=active")

# ── 3. Get group detail with legs ──
print(f"\n🔷 Getting group {group_id} detail...")
test("get", f"/groups/{group_id}")

# ── 4. Add a third leg ──
print(f"\n🔷 Adding a 3rd leg to group {group_id}...")
r2 = test("post", f"/groups/{group_id}/legs", json={
    "symbol": "GOLDM26MAR143000PE",
    "exchange": "MCX",
    "quantity": 1,
    "transaction_type": "BUY",
    "entry_price": 890.0,
    "product": "NRML"
})
leg_id = r2.json().get("leg_id")

# ── 5. List legs for the group ──
print(f"\n🔷 Listing legs for group {group_id}...")
test("get", f"/groups/{group_id}/legs")

# ── 6. Remove the 3rd leg ──
print(f"\n🔷 Removing leg {leg_id}...")
test("delete", f"/legs/{leg_id}")

# ── 7. Verify leg removed ──
print(f"\n🔷 Verifying legs (should be 2 active)...")
test("get", f"/groups/{group_id}/legs?status=active")

# ── 8. Check order log ──
print(f"\n🔷 Checking order log...")
test("get", "/orders?limit=10")

# ── 9. Close the group ──
print(f"\n🔷 Closing group {group_id}...")
test("delete", f"/groups/{group_id}")

# ── 10. Verify group closed ──
print(f"\n🔷 Verifying group is closed...")
test("get", f"/groups/{group_id}")

print("\n✅ All group CRUD tests complete!")
