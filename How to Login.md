# How to Login - Kite Connect Authentication Guide

This guide explains how to authenticate with the Zerodha Kite Connect API using the provided authentication scripts.

## Overview

The authentication process uses two main scripts:
1. **`get_token_local.py`** - Generates and saves access tokens via OAuth flow
2. **`load_kite_from_access.py`** - Loads saved tokens and creates an authenticated Kite Connect session

## Prerequisites

Before starting, ensure you have:

1. **Zerodha Kite Connect App**
   - Create an app at [Kite Connect Developer Console](https://developers.kite.trade/apps)
   - Note your `api_key` and `api_secret`
   - Set redirect URL to `http://127.0.0.1:5010`

2. **Required Configuration Files**
   - `secrets/.env` file with your credentials:
     ```env
     KITE_API_KEY=your_api_key_here
     KITE_API_SECRET=your_api_secret_here
     ```

3. **Python Packages**
   ```bash
   pip install kiteconnect python-dotenv
   ```

## Authentication Process

### Step 1: Generate Access Token

Run `get_token_local.py` to start the OAuth authentication flow:

```bash
python get_token_local.py
```

**What happens:**
1. A local HTTP server starts on port 5010
2. Your browser opens with the Kite Connect login page
3. Log in with your Zerodha credentials and authorize the app
4. You'll be redirected to `http://127.0.0.1:5010` with a request token
5. The script exchanges the request token for an access token
6. Tokens are saved to `secrets/.env_access` with a timestamp

**Output:**
```
[INFO] Starting local server on port 5010...
[INFO] Opening browser for authentication...
[INFO] Access Token received: abcd1...
[INFO] Saving tokens to: C:\path\to\secrets\.env_access
✓ Tokens saved successfully at: 2026-02-01 14:30:45
```

**Token Details Saved:**
- `KITE_REQUEST_TOKEN` - One-time token from OAuth redirect
- `KITE_ACCESS_TOKEN` - Valid for 24 hours
- `KITE_TOKEN_SAVED_TIME` - Timestamp for tracking token age

### Step 2: Use Authenticated Session

Once tokens are saved, use `load_kite_from_access.py` to create authenticated sessions:

```python
import load_kite_from_access

# Get authenticated Kite Connect instance
kite = load_kite_from_access.kite

# Use the kite object for API calls
positions = kite.positions()
orders = kite.orders()
```

**What it does:**
- Loads `KITE_ACCESS_TOKEN` from `secrets/.env_access`
- Creates a `KiteConnect` instance with your API key
- Sets the access token for authentication
- Returns a ready-to-use `kite` object

## Token Expiry & Refresh

⚠️ **Important**: Access tokens expire daily at market close (around 3:30 PM IST).

**To refresh:**
1. Run `get_token_local.py` again
2. Complete the login flow
3. New tokens will overwrite old ones in `secrets/.env_access`

**No need to:**
- Delete old tokens manually
- Restart your application (if using module reload - see test script below)

## Testing Your Authentication

Use this script to verify your authentication is working:

```python
# Reload the module to get fresh kite object with new token
import sys

# Remove cached module to force fresh import
if 'load_kite_from_access' in sys.modules:
    del sys.modules['load_kite_from_access']
    print("🔄 Cleared cached load_kite_from_access module")

import load_kite_from_access
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
    except Exception as e:
        print(f"Error getting profile: {e}")
else:
    print("\n❌ Kite object is None - session not established")
    print("Please ensure you have run get_token_local.py to generate access token")
```

**Expected Output (Success):**
```
🔄 Cleared cached load_kite_from_access module
Kite object: <kiteconnect.connect.KiteConnect object at 0x...>
Type: <class 'kiteconnect.connect.KiteConnect'>

✅ UserID: AB1234
✅ User Name: Your Name
✅ Email: your.email@example.com
```

**Expected Output (Failure):**
```
Error getting profile: Incorrect `api_key` or `access_token`.
```
→ **Solution**: Run `get_token_local.py` to generate a fresh token

## Troubleshooting

### "Insufficient permissions" Error
- **Cause**: Your Kite Connect app doesn't have market data permissions or your data subscription is inactive
- **Solution**: 
  - Check your app permissions at [Kite Connect Console](https://developers.kite.trade/apps)
  - Verify NSE/MCX data subscriptions in Kite Web → Account → Funds

### Port 5010 Already in Use
- **Cause**: Another process is using port 5010
- **Solution**: 
  - Kill the process using the port, or
  - Modify the port in `get_token_local.py` (update both server and redirect URL)

### "Incorrect api_key or access_token" Error
- **Cause**: Token expired or invalid credentials
- **Solution**: Run `get_token_local.py` to get a fresh token

### Browser Doesn't Open Automatically
- **Cause**: System browser configuration or permissions
- **Solution**: Manually navigate to `http://127.0.0.1:5010` after starting the script

## File Structure

```
pykite/
├── get_token_local.py          # OAuth token generation
├── load_kite_from_access.py    # Session creation
├── config.py                   # Configuration utilities
└── secrets/
    ├── .env                    # API credentials (api_key, api_secret)
    └── .env_access             # Saved tokens (auto-generated)
```

## Security Notes

🔒 **Keep these files private:**
- `secrets/.env` - Contains your API key and secret
- `secrets/.env_access` - Contains your active access token

⚠️ **Do NOT commit to Git:**
- Add `secrets/` to your `.gitignore`
- Never share your `api_key`, `api_secret`, or `access_token`

## Additional Resources

- [Kite Connect Documentation](https://kite.trade/docs/connect/v3/)
- [Zerodha Developer Console](https://developers.kite.trade/apps)
- [KiteConnect Python Library](https://github.com/zerodhatech/pykiteconnect)

---

**Last Updated**: February 2026
