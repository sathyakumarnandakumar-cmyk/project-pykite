# Kite Deployment Package

This package contains tools for connecting to Zerodha Kite Connect API, managing login sessions, and running automated trading logic (like Trailing Stop Loss).

## 🚀 Quick Start

### 1. Setup Secrets
or security, credentials are **not** stored in this repo. You must create them manually.

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

---

## 📚 API Research Notebook Functions

The `api_research.ipynb` notebook comes pre-loaded with several helper functions and classes for trading automation.

### 🔹 Helper Functions

#### `get_orders(kite)`
- **Purpose**: Fetches and prints all orders for the day.
- **Returns**: List of order dictionaries.

#### `get_holdings(kite)`
- **Purpose**: Fetches and prints current portfolio holdings.
- **Returns**: List of holding dictionaries.

#### `get_positions(kite)`
- **Purpose**: Fetches and prints current positions (net and day).
- **Returns**: Dictionary with 'net' and 'day' keys.

#### `place_stop_loss_gtt(kite, tradingsymbol, exchange, trigger_percent, quantity, product)`
- **Purpose**: Places a **Single GTT Sell Order** to act as a Stop Loss.
- **Logic**:
  1. Fetches current LTP.
  2. Calculates trigger price based on simple percentage drop (LTP * (1 - percent)).
  3. Places a GTT order on the broker server.
- **Use Case**: Setting an initial stop loss that doesn't need monitoring.

---

### 🔹 Advanced Classes

#### `PriceMonitor`
**Purpose**: Efficiently fetches prices for multiple symbols in a single API call (bulk fetch), decoupling price updates from logic.

- **`add_symbol(exchange, symbol)`**: Adds a stock to the watchlist.
- **`update_prices()`**: Fetches fresh prices for **ALL** watchlist symbols in one go.
- **`get_price(exchange, symbol)`**: Returns the last fetched price from memory.

#### `TrailingStopLossManager`
**Purpose**: Implements Trailing Stop Loss logic using GTT orders and a persistent disk cache.

- **Storage**: Uses `tsl_cache.json` to remember High Water Marks (HWM) even if the notebook is restarted.
- **`update_trailing_stop_loss(...)`**:
  - Updates HWM if price moves up.
  - Calculates new SL price.
  - **Modifies** the existing GTT order only if the new SL is higher than the current trigger.
- **`update_bulk_trailing_stop_loss(positions_list)`**:
  - Optimized version that fetches all prices first and then updates all positions efficiently.

---

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
