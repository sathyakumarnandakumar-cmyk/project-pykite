import os
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

# Import centralized config
from config import ENV_FILE, ENV_ACCESS_FILE, API_KEY

def get_kite_session():
    """
    Loads credentials and returns a validated KiteConnect instance.
    """
    # 1. Load from secrets/.env_access
    if not load_dotenv(dotenv_path=ENV_ACCESS_FILE, override=True):
        print(f"❌ Error: Could not load {ENV_ACCESS_FILE}")
        return None

    api_key = API_KEY
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not api_key or not access_token:
        print("❌ Error: Missing KITE_API_KEY or KITE_ACCESS_TOKEN in file.")
        return None

    # 2. Initialize KiteConnect
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    try:
        # 3. Validate session
        user = kite.profile()
        print(f"✅ Connected: {user['user_name']} (ID: {user['user_id']})")
        return kite
    except Exception as e:
        print(f"⚠️ Session Invalid: {e}")
        return None

# --- EXPORTABLE INSTANCE ---
# This allows other files to use 'from kite_loader import kite'
kite = get_kite_session()

# --- STANDALONE TEST ---

# 3. Initialize KiteConnect session   


if __name__ == "__main__":
    print("\n--- Running Standalone Session Test ---")
    if kite:
        print("Session is active and ready for use.")
        # Example: Print margins to prove it works
        margins = kite.margins()
        print(f"Available Margin is: {margins['equity']['net']}")
    else:
        print("Failed to initialize Kite session.")
