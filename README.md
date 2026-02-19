# Kite Deployment Package

This package contains tools for connecting to Zerodha Kite Connect API, managing login sessions, and running automated trading logic (like Trailing Stop Loss).

## 🚀 Quick Start

### 1. Setup Secrets
For security, credentials are **not** stored in this repo. You must create them manually.

1.  Create a folder named `secrets` inside this directory:
    ```bash
    mkdir secrets
    ```
2.  Create a file named `.env` inside `secrets/`:
    ```bash
    # secrets/.env content
    KITE_API_KEY=your_api_key_here
    KITE_API_SECRET=your_api_secret_here
    KITE_USER_ID=your_zerodha_user_id
    KITE_PASSWORD=your_zerodha_password
    KITE_REDIRECT_URI=http://127.0.0.1:5010/login  # Default
    ```
3.  The request/access tokens will be automatically saved to `secrets/.env_access` by the login script.

### 2. Login
Run the login test script to authenticate and generate an access token:
```bash
python login_test.py
```
This will:
- Open the login URL in your browser if needed.
- Capture the redirect and save the token.
- Verify your profile.

### 3. Run Research / TSL Logic
Use the Jupyter Notebook `api_research.ipynb` to run your strategies.

- **Trailing Stop Loss (TSL)**:
  The notebook contains a `TrailingStopLossManager` class that:
  - Tracks High Water Mark (HWM) in `tsl_cache.json`.
  - Places/Modifies GTT orders automatically.
  - Updates in bulk for multiple positions.

## 📂 File Structure

- `api_research.ipynb`: Main notebook for running TSL and research.
- `config.py`: Central configuration loading credentials from `secrets/`.
- `get_token_local.py`: Handles the OAuth login flow and saves tokens.
- `load_kite_from_access.py`: Utility to load an authenticated `kite` object.
- `login_test.py`: Script to verify login and profile fetching.
- `main_fastapi.py`: Optional FastAPI server for cloud deployment.
- `secrets/`: **[GIT IGNORED]** Stores your keys and tokens.

## ⚠️ Important
- Never commit the `secrets/` folder to GitHub.
- The `.gitignore` file is already set up to exclude it.
