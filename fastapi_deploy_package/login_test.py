"""
This script replicates the logic from tests.ipynb (cells 1-6) to:
1. Initialize Kite Connect session
2. Check existing token validity
3. Trigger login flow if needed (using get_token_local)
4. Verify profile information

End goal: Adaptable for Cloud Run / FastAPI deployment.
"""
import sys
import importlib
from pprint import pprint
import pandas as pd
import numpy as np
from datetime import datetime, date

# Import local modules
import config
import get_token_local
import load_kite_from_access

def login_kite():
    """Triggers the local login flow (v2)"""
    print("Starting login sequence...")
    try:
        # Reloading modules to ensure fresh state if running interactively
        importlib.reload(config)
        importlib.reload(get_token_local)
        
        if not hasattr(get_token_local, 'login'):
            print("Error: get_token_local.login not found.")
            return
        
        get_token_local.login()
    except Exception as e:
        print(f"Error during login: {e}")

def get_kite_profile():
    print("--- Kite Connect Login Test ---")
    
    # Reload the module to get fresh kite object with new token (if needed)
    if 'load_kite_from_access' in sys.modules:
        importlib.reload(load_kite_from_access)
        print("🔄 Reloaded load_kite_from_access module")

    kite = load_kite_from_access.kite
    
    # Check kite status and get UserID
    print(f"Kite object: {kite}")
    print(f"Type: {type(kite)}")
    
    if kite:
        try:
            profile = kite.profile()
            print(f"\n✅ UserID: {profile['user_id']}")
            print(f"✅ User Name: {profile['user_name']}")
            print(f"✅ Email: {profile['email']}")
            print(f"✅ Full Profile: {profile}")
            return profile
        except Exception as e:
            print(f"Error getting profile: {e}")
            print("Attempting to re-login...")
            login_kite()
            # After login, try to reload again
            importlib.reload(load_kite_from_access)
            kite = load_kite_from_access.kite
            if kite:
                try:
                    profile = kite.profile()
                    print(f"\n✅ [Retry] UserID: {profile['user_id']}")
                    return profile
                except Exception as e:
                     print(f"Still failed after login: {e}")
                     return None

    else:
        print("\n❌ Kite object is None - session not established")
        print("Initiating login sequence...")
        login_kite()
        # After login, try to reload again
        importlib.reload(load_kite_from_access)
        kite = load_kite_from_access.kite
        if kite:
             try:
                profile = kite.profile()
                print(f"\n✅ [Retry] UserID: {profile['user_id']}")
                return profile
             except Exception as e:
                 print(f"Still failed after login: {e}")
                 return None
    return None


def main():
    print("--- Kite Connect Login Test ---")
    get_kite_profile()

if __name__ == "__main__":
    main()
