# Detailed Analysis of `api_research.ipynb`

This document provides a highly elaborate, step-by-step breakdown of all cells, imports, functions, logical flows, state management, and executions present in the `api_research.ipynb` notebook located in the `deploy` folder. The notebook acts as a research, prototyping, and testing environment for the Kite Connect API and algorithm development, specifically focusing on Trailing Stop Loss (TSL) mechanics.

---

## 1. Environment Setup & Imports
To facilitate testing within the project structure, the notebook ensures it can properly import required packages from its context.

**Execution Flow:**
- Custom path configuration: It retrieves the current working directory (`os.getcwd()`) and appends it to `sys.path`. This guarantees that local modules from the `deploy` directory can be successfully imported without `ModuleNotFoundError`.
- **Local Module Imports:**
  - `config`: Presumed to contain API credentials and constants.
  - `login_test`: Tested immediately after setup; used to verify the Kite connection state.
  - `main_fastapi`: Standard FastAPI deployment configurations (imported for context checks but unused directly in this draft).
  - `load_kite_from_access`: A dedicated module responsible for loading the initialized `KiteConnect` broker object instance from cached access tokens. Fast-tracks the authentication flow without requiring manual GUI login.

---

## 2. Basic API Connection Executions & Testing
The notebook systematically verifies the connection state using multiple independent checks.

### Fetching User Profile Execution
- Executes `login_test.get_kite_profile()`.
- Implements a `try-except` block to gracefully handle scenarios where the session has expired or no token exists.
- **Success:** Prints `Profile fetched successfully: [user_name]` to standard output.
- **Failure:** Advises the analyst to "Try hitting the login endpoint first."

### The "Scratchpad" Execution
- Actively imports the configured `load_kite_from_access` script.
- Assigns the `kite` connect object from `load_kite_from_access.kite` to a local `kite` variable, which serves as the primary gateway for all subsequent API requests.
- Prints the object to test if instantiation was successful.

---

## 3. Account Data Helper Functions
A set of dedicated wrapper functions designed to fetch current broker account data. They are designed with resilient error handling so the execution does not break in the event of an API error.

### `get_orders(kite)`
- **Behavior:** Queries the API (`kite.orders()`) for the day's order book.
- **Returns:** Returns an array of order dictionaries, or `None` if an `Exception` is caught.
- **Output:** Logs "Fetched `X` orders."

### `get_holdings(kite)`
- **Behavior:** Queries the API (`kite.holdings()`) for the user's permanent holdings (T1/CNC assets).
- **Returns:** Returns an array of holding dictionaries, or `None` if an `Exception` is caught. 
- **Output:** Logs "Fetched `X` holdings."

### `get_positions(kite)`
- **Behavior:** Queries the API (`kite.positions()`) for intraday and overnight derivatives positions.
- **Returns:** A dictionary mapping with two main keys: `net` and `day`, corresponding to net positions and purely intraday positions respectively, or `None` on error.
- **Output:** Logs "Fetched positions: `dict_keys(...)`"

---

## 4. Operational Trading Functions: The GTT Engine
GTT (Good Till Triggered) orders are a system provided by Zerodha that lives on their servers until a specific price point is breached.

### `place_stop_loss_gtt(kite, tradingsymbol, exchange, trigger_percent, quantity=1, product=None)`
- **Purpose:** Algorithmically constructs and places a Single-Leg GTT Sell order acting as a stop-loss, positioned at a specified generic percentage below the current market price.
- **Arguments:**
  - `kite`: Authorized API instance.
  - `tradingsymbol`: Asset identifier (e.g., `'INFY'`).
  - `exchange`: Market exchange (e.g., `'NSE'`, `'BSE'`, `'MCX'`).
  - `trigger_percent`: Strict positive integer or float representing the % dropdown threshold (e.g., `5` means 5%).
  - `quantity`: Executable lot size.
  - `product`: Explicit CNC/NRML product type string (optional fallback logic provided).
- **Elaborate Execution Logic:**
  1. **LTP Fetch:** Requests the live quote (Last Traded Price) via `kite.ltp('EXCHANGE:SYMBOL')`. It aggressively validates the response to ensure `last_price` exists.
  2. **Trigger Computation:** 
     - Evaluates `last_price * (1 - (trigger_percent / 100))`. 
     - **Crucial Rounding:** Applies tick size rounding (`round(trigger_price * 20) / 20`) to map the mathematical float to the nearest valid tradable tick `0.05` resolution on NSE.
  3. **Order Preparation:**
     - Determines `trigger_type = kite.GTT_TYPE_SINGLE`.
     - Checks if `product` was supplied; if not, defaults to `kite.PRODUCT_CNC` for NSE assets and `kite.PRODUCT_NRML` for others.
     - Creates the nested execution order payload specifying `TRANSACTION_TYPE_SELL` and `ORDER_TYPE_LIMIT`. (Note: The limit price is equivalent to the trigger price).
  4. **Placement:** Fires off `kite.place_gtt()`. Returns the generated unique `gtt_id`, or `None` upon catch exception. Prints verbose status messages.

---

## 5. Architectural Classes for Trailing Stop Loss Management
The core architectural prototype developed in this notebook involves a split-concern approach to trailing stop loss: a lightweight price fetcher, and state-driven tracker. 

### `PriceMonitor` Class
**Purpose:** Acts as a centralized background service that batches LTP requests into a single unified API call for multiple instruments. This drastically curtails the number of distinct API hits overhead rate limits.
- **Properties:**
  - `self.kite`: Kite object cache.
  - `self.watchlist`: A python `set()` containing universally unique strings like `'NSE:INFY'`.
  - `self.latest_prices`: A hash map (dictionary) tracking the most recently fetched price per key.
- **Methods:**
  - `add_symbol(...)`: Formats the `exchange:symbol` key and registers it in `self.watchlist`.
  - `remove_symbol(...)`: Dynamically clears a key from `self.watchlist` and deletes the value inside `self.latest_prices` if it exists.
  - `update_prices(self)`: The main polling execution. Formats the `watchlist` set into a list object and executes a single bulk `kite.ltp()` retrieval. It then mutates `self.latest_prices` with the fresh dict values. Features generic exception handling mapping to prevent routine background thread crashes.
  - `get_price(...)`: Contains fail-proof fetching via python dict's `.get()` strategy, returning data without raising `KeyError`.

### `TrailingStopLossManager` Class
**Purpose:** Owns the business logic of pushing limits up based on dynamic price action and handles disk persistence so tracking can survive temporary script restarts or crashes.
- **Properties:**
  - `self.kite`: Access pointer to the broker instance.
  - `self.cache_file`: Points to a local disk storage location (`tsl_cache.json`).
  - `self.cache`: Active memory dict mapping derived from `load_cache()`.
- **Methods:**
  - `load_cache(self)`: Safe JSON parsing wrapped in `os.path.exists()` and exception handlers. Fails gracefully to a pristine `{}`.
  - `save_cache(self)`: Synchronously dumps the active memory `self.cache` format directly back down into `tsl_cache.json` utilizing `indent=4` for human readability.
  - `update_trailing_stop_loss(self, tradingsymbol, exchange, trailing_percent, quantity, product, gtt_id, current_ltp)`: The primary worker routine.
    - **Initial Check:** Re-fetches the current LTP on its own if `current_ltp` (meant to be fed by the `PriceMonitor`) is surprisingly `None`. Emits a performance warning when doing so.
    - **Cache Seeding:** Evaluates if the `instrument_key` isn't memory resident yet. If it misses, it builds an instantiation profile injecting `hwm_price = current_ltp` (High Water Mark) and defaults `last_trigger_price` to `0`.
    - **Continuous State Tracking:**
      - **HWM Update Routine:** Examines if the incoming `current_ltp` exceeds the tracked `hwm_price`. If yes, it decisively updates the HWM variable to the new threshold limit and instantly calls `save_cache()` to assure safety against immediately ensuing crash vectors.
    - **Trailing Re-evaluation:**
      - Mathematically resolves the `new_sl_price` relative purely to the `hwm_price` leveraging the identical integer drop / tick rounding (0.05 step) formula depicted in `place_stop_loss_gtt`.
    - **Action Logic Execution:**
      - Tests condition `if new_sl_price > current_trigger_price:`. Only processes on upward trailing dynamics.
      - **Scenario A (First Trace):** No `gtt_id` is tracked. Consequently invokes the auxiliary function `place_stop_loss_gtt(...)`, records the returned ID code internally to memory alongside the latest trigger price, and saves.
      - **Scenario B (Progression Trail):** Utilizes the `kite.modify_gtt(...)` payload API overriding the current ID utilizing `new_sl_price` data arrays. Contains specialized error handling: If the API spits back that a GTT ID invalidation occurred (`"not found"` / `InputException` indicating perhaps manual cancellation or broker clearance), it aggressively scrubs the memory `entry["gtt_id"] = None` forcing Scenario A recreation upon the subsequent loop tick. 

---

## 6. End-to-End Local Testing Block
The conclusion of the notebook constitutes a functional prototype testing space where theory turns to execution.

1. **Basic Queries Execution:** Executes `get_orders`, `get_holdings`, and `get_positions`, printing slice lengths (`orders[:1]`) and dictionary key sizes (`len(positions.get('net', []))`) to output validation visually.
2. **Infrastructure Hookup:** 
   - Invokes `monitor = PriceMonitor(kite)`
   - Invokes `tsl_manager = TrailingStopLossManager(kite)`
3. **Mock Environment Loading:** 
   - Spins up a manual `my_positions` list referencing fake data: INFY (NSE, 5%, 10 qty) and TCS (NSE, 4%, 5 qty).
   - Commits iterators loading both items directly into `monitor.add_symbol()`.
4. **Mock Loop Step Execution:** 
   - Signals a singular "simulated" run iteration. 
   - Manually blasts `monitor.update_prices()`, firing off exactly one batch quote poll across the Kite API. 
   - Operates a `for` loop dynamically reading back from `monitor.get_price(...)`. 
   - Conditionally pushes fetched live data directly downstream into `tsl_manager.update_trailing_stop_loss(...)`. 
   - This exact configuration verifies the seamless data flow linkage between asynchronous multi-price background tracking and rigid serial algorithmic logic execution mechanisms within a singular loop timeframe.
