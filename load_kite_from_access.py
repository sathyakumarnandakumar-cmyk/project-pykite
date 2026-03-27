import os
import sys
from kiteconnect import KiteConnect

# Since this file is in deploy/, we import config directly
from config import get_app_config, load_access_token

def get_kite_session(app_name="Full_App"):
    """
    Loads credentials for a SPECIFIC APP and returns a validated KiteConnect instance.
    """
    try:
        # 1. Fetch config for this specific app
        app_conf = get_app_config(app_name)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return None

    api_key = app_conf.get("KITE_API_KEY")
    
    # 2. Look for the app-specific access token in access.ini
    access_token = load_access_token(app_name)

    if not api_key or not access_token:
        print(f"❌ Error: Missing API_KEY or KITE_ACCESS_TOKEN for {app_name}.")
        print(f"Have you run the login flow for {app_name} yet?")
        return None

    # 3. Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    try:
        # 4. Validate session
        user = kite.profile()
        print(f"✅ [{app_name}] Connected: {user['user_name']} (ID: {user['user_id']})")
        return kite
    except Exception as e:
        print(f"⚠️ [{app_name}] Session Invalid: {e}")
        return None

# --- STANDALONE TEST ---
if __name__ == "__main__":
    app_to_test = sys.argv[1] if len(sys.argv) > 1 else "Full_App"
    print(f"\n--- Running Standalone Session Test for {app_to_test} ---")
    print(f"ℹ️ You can test other apps like: `python deploy/load_kite_from_access.py algotestlive`")
    
    test_kite = get_kite_session(app_to_test)
    if test_kite:
        print("Session is active and ready for use.")
        margins = test_kite.margins()
        print(f"Available Margin is: {margins['equity']['net']}")
    else:
        print("Failed to initialize Kite session. You may need to login first.")
