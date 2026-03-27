"""
Centralized configuration for Kite Connect credentials and paths based on INI architecture.
All secrets are loaded from secrets/secrets.ini using configparser.
"""
import os
import configparser
from datetime import datetime

# Base directory (pykite folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Note: config.py is in 'deploy' so pykite is parent
SECRETS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'secrets')

if not os.path.exists(SECRETS_DIR):
    # Fallback if config.py is moved to root
    SECRETS_DIR = os.path.join(BASE_DIR, 'secrets')

# Environment file paths
SECRETS_INI = os.path.join(SECRETS_DIR, 'secrets.ini')
ACCESS_INI = os.path.join(SECRETS_DIR, 'access.ini')

# Initialize and read secrets.ini (preserve case)
_config = configparser.ConfigParser()
_config.optionxform = str
if os.path.exists(SECRETS_INI):
    _config.read(SECRETS_INI)
else:
    print(f"⚠ Warning: {SECRETS_INI} not found!")

def get_app_config(app_name: str) -> dict:
    """
    Returns the configuration dictionary for a specific app.
    Raises ValueError if the app is not found in secrets.ini.
    """
    if not _config.has_section(app_name):
        raise ValueError(f"App '{app_name}' not found in {SECRETS_INI}.")
    return dict(_config[app_name])


def save_tokens(app_name: str, request_token: str, access_token: str):
    """Save request_token and access_token to secrets/access.ini under the specific app section"""
    print(f"[INFO] Saving tokens to: {os.path.abspath(ACCESS_INI)}")
    
    # Read existing access.ini or create new one
    access_config = configparser.ConfigParser()
    access_config.optionxform = str
    if os.path.exists(ACCESS_INI):
        access_config.read(ACCESS_INI)
    
    if not access_config.has_section(app_name):
        access_config.add_section(app_name)
        
    saved_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    access_config.set(app_name, "KITE_REQUEST_TOKEN", request_token)
    access_config.set(app_name, "KITE_ACCESS_TOKEN", access_token)
    access_config.set(app_name, "KITE_TOKEN_SAVED_TIME", saved_time)
    
    with open(ACCESS_INI, 'w') as configfile:
        access_config.write(configfile)
        
    print(f"✓ Tokens saved successfully for {app_name} at: {saved_time}")


def load_access_token(app_name: str) -> str:
    """Load access_token from secrets/access.ini for a specific app"""
    access_config = configparser.ConfigParser()
    access_config.optionxform = str
    if os.path.exists(ACCESS_INI):
        access_config.read(ACCESS_INI)
        if access_config.has_section(app_name):
            return access_config.get(app_name, "KITE_ACCESS_TOKEN", fallback=None)
    return None


def validate_credentials(app_name: str) -> bool:
    """Check if all required credentials are set for the given app"""
    try:
        conf = get_app_config(app_name)
    except ValueError as e:
        print(f"⚠ Validation failed: {e}")
        return False

    missing = []
    if not conf.get('KITE_API_KEY'):
        missing.append('KITE_API_KEY')
    if not conf.get('KITE_API_SECRET'):
        missing.append('KITE_API_SECRET')
    if not conf.get('KITE_USER_ID'):
        missing.append('KITE_USER_ID')
    if not conf.get('KITE_PASSWORD'):
        missing.append('KITE_PASSWORD')
    
    if missing:
        print(f"⚠ Missing credentials for {app_name}:")
        for var in missing:
            print(f"  - {var}")
        return False
    return True


# Debug info (only when run directly)
if __name__ == "__main__":
    print(f"SECRETS_DIR: {SECRETS_DIR}")
    print(f"SECRETS_INI: {SECRETS_INI}")
    print(f"ACCESS_INI: {ACCESS_INI}")
    
    for section in _config.sections():
        print(f"\n--- Testing App: {section} ---")
        conf = get_app_config(section)
        api_key = conf.get('KITE_API_KEY')
        api_secret = conf.get('KITE_API_SECRET')
        user_id = conf.get('KITE_USER_ID')
        pwd = conf.get('KITE_PASSWORD')
        
        print(f"  API_KEY: {api_key[:5]}..." if api_key else "  API_KEY: None")
        print(f"  API_SECRET: {api_secret[:5]}..." if api_secret else "  API_SECRET: None")
        print(f"  USER_ID: {user_id}")
        print(f"  PASSWORD: {'*' * len(pwd) if pwd else 'None'}")
        print(f"  Validation: {'✓ OK' if validate_credentials(section) else '✗ FAILED'}")
