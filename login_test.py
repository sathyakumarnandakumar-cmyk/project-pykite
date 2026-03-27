"""
This script replicates the logic to:
1. Initialize Kite Connect session
2. Check existing token validity
3. Trigger login flow if needed (using get_token_local)
4. Verify profile information

Now updated to test specific apps under the INI Architecture.
"""
import sys
import os
import importlib

# Ensure we can import from the parent directory if run sequentially
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import get_token_local
import load_kite_from_access

def login_kite(app_name: str):
    """Triggers the local login flow for the specific app"""
    print(f"Starting login sequence for {app_name}...")
    try:
        # Reloading modules to ensure fresh state if running interactively
        importlib.reload(config)
        importlib.reload(get_token_local)
        
        get_token_local.login(app_name)
    except Exception as e:
        print(f"Error during login: {e}")

def get_kite_profile(app_name: str):
    print(f"\n--- Kite Connect Login Test: {app_name} ---")
    
    # Reload the module to get fresh tokens
    if 'load_kite_from_access' in sys.modules:
        importlib.reload(load_kite_from_access)
        print("🔄 Reloaded load_kite_from_access module")

    kite = load_kite_from_access.get_kite_session(app_name)
    
    if kite:
        try:
            profile = kite.profile()
            print(f"\n✅ UserID: {profile['user_id']}")
            print(f"✅ User Name: {profile['user_name']}")
            print(f"✅ Email: {profile['email']}")
            return profile
        except Exception as e:
            print(f"Error getting profile: {e}")
            print("Attempting to re-login...")
            login_kite(app_name)
            
            # After login, try to reload again
            importlib.reload(load_kite_from_access)
            kite = load_kite_from_access.get_kite_session(app_name)
            if kite:
                try:
                    profile = kite.profile()
                    print(f"\n✅ [Retry] UserID: {profile['user_id']}")
                    return profile
                except Exception as ex:
                     print(f"Still failed after login: {ex}")
                     return None
    else:
        print(f"\n❌ Kite object is None - session not established for {app_name}")
        print("Initiating login sequence...")
        login_kite(app_name)
        
        # After login, try to reload again
        importlib.reload(load_kite_from_access)
        kite = load_kite_from_access.get_kite_session(app_name)
        if kite:
             try:
                profile = kite.profile()
                print(f"\n✅ [Retry] UserID: {profile['user_id']}")
                return profile
             except Exception as ex:
                 print(f"Still failed after login: {ex}")
                 return None
    return None

def main():
    # Prompt the user or default to Full_App
    if len(sys.argv) > 1:
        app_to_test = sys.argv[1]
    else:
        app_to_test = "Full_App"
        print(f"ℹ️ Testing '{app_to_test}' by default. You can pass an app name like: `python login_test.py algotestlive`")
    
    get_kite_profile(app_to_test)

if __name__ == "__main__":
    main()
