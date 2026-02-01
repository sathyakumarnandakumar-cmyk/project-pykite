"""
Centralized configuration for Kite Connect credentials and paths.
All secrets are loaded from pykite/secrets/.env
All scripts should import from this module.
"""
import os
from dotenv import load_dotenv, set_key

# Base directory (pykite folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRETS_DIR = os.path.join(BASE_DIR, 'secrets')

# Environment file paths
ENV_FILE = os.path.join(SECRETS_DIR, '.env')
ENV_ACCESS_FILE = os.path.join(SECRETS_DIR, '.env_access')

# Load environment variables from secrets/.env
if os.path.exists(ENV_FILE):
    load_dotenv(ENV_FILE, override=True)
else:
    print(f"⚠ Warning: {ENV_FILE} not found!")

# --- KITE CREDENTIALS ---
API_KEY = os.environ.get('KITE_API_KEY')
API_NAME = os.environ.get('KITE_API_NAME')
API_SECRET = os.environ.get('KITE_API_SECRET')
USER_ID = os.environ.get('KITE_USER_ID')
PASSWORD = os.environ.get('KITE_PASSWORD')
REDIRECT_URI = os.environ.get('KITE_REDIRECT_URI', 'http://127.0.0.1:5010/login')

# --- OTHER CONFIG ---
PORT = 5010
LOGIN_URL = f"https://kite.zerodha.com/connect/login?v=3&api_key={API_KEY}" if API_KEY else None


def save_tokens(request_token: str, access_token: str):
    """Save request_token and access_token to secrets/.env_access"""
    from datetime import datetime
    
    print(f"[INFO] Saving tokens to: {os.path.abspath(ENV_ACCESS_FILE)}")
    
    # Save tokens with timestamp
    saved_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    set_key(ENV_ACCESS_FILE, "KITE_REQUEST_TOKEN", request_token)
    set_key(ENV_ACCESS_FILE, "KITE_ACCESS_TOKEN", access_token)
    set_key(ENV_ACCESS_FILE, "KITE_TOKEN_SAVED_TIME", saved_time)
    
    print(f"✓ Tokens saved successfully at: {saved_time}")


def load_access_token() -> str:
    """Load access_token from secrets/.env_access"""
    if os.path.exists(ENV_ACCESS_FILE):
        load_dotenv(ENV_ACCESS_FILE, override=True)
        return os.environ.get('KITE_ACCESS_TOKEN')
    return None


def validate_credentials():
    """Check if all required credentials are set"""
    missing = []
    if not API_KEY:
        missing.append('KITE_API_KEY')
    if not API_SECRET:
        missing.append('KITE_API_SECRET')
    if not USER_ID:
        missing.append('KITE_USER_ID')
    if not PASSWORD:
        missing.append('KITE_PASSWORD')
    
    if missing:
        print("⚠ Missing environment variables:")
        for var in missing:
            print(f"  - {var}")
        return False
    return True


# Debug info (only when run directly)
if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"SECRETS_DIR: {SECRETS_DIR}")
    print(f"ENV_FILE: {ENV_FILE}")
    print(f"ENV_ACCESS_FILE: {ENV_ACCESS_FILE}")
    print(f"\nCredentials loaded:")
    print(f"  API_KEY: {API_KEY[:5]}..." if API_KEY else "  API_KEY: None")
    print(f"  API_NAME: {API_NAME}")
    print(f"  API_SECRET: {API_SECRET[:5]}..." if API_SECRET else "  API_SECRET: None")
    print(f"  USER_ID: {USER_ID}")
    print(f"  PASSWORD: {'*' * len(PASSWORD) if PASSWORD else 'None'}")
    print(f"\nValidation: {'✓ OK' if validate_credentials() else '✗ FAILED'}")
