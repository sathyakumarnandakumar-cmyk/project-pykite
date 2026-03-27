"""
strategy_group.py — Composite strategy grouping for multi-leg positions.

Groups multiple position legs (e.g. Future + Short Call = Synthetic Put,
or Short Call + Short Put = Short Straddle) and provides group-level
trailing stop-loss management.

Trigger modes:
  - combined_pnl : SL fires when combined unrealized P&L drops X% from HWM
  - underlying   : SL fires when a reference underlying drops X% from HWM

Tables created:
  - strategy_groups       : one row per composite strategy
  - strategy_group_legs   : legs belonging to each group
  - order_log             : audit trail of every order placed (individual + group)
"""

import sqlite3
import os
import time
from datetime import datetime


# ─────────────────────────────────────────────
#  DB Schema Initialisation
# ─────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategy_groups (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    trigger_mode     TEXT NOT NULL DEFAULT 'combined_pnl',   -- 'combined_pnl' | 'underlying'
    ref_symbol       TEXT,          -- underlying symbol  (for trigger_mode='underlying')
    ref_exchange     TEXT,          -- underlying exchange (for trigger_mode='underlying')
    trailing_percent REAL NOT NULL,
    hwm_value        REAL NOT NULL DEFAULT 0,
    last_trigger_value REAL DEFAULT 0,
    status           TEXT NOT NULL DEFAULT 'active',
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS strategy_group_legs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id         INTEGER NOT NULL REFERENCES strategy_groups(id),
    symbol           TEXT NOT NULL,
    exchange         TEXT NOT NULL,
    quantity         INTEGER NOT NULL,
    transaction_type TEXT NOT NULL,      -- 'BUY' or 'SELL'
    entry_price      REAL,
    product          TEXT DEFAULT 'NRML',
    gtt_id           INTEGER,            -- individual GTT for this leg (if any)
    status           TEXT NOT NULL DEFAULT 'active',
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS order_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source           TEXT NOT NULL,       -- 'individual' | 'group_exit' | 'tsl_gtt' | 'manual'
    source_id        INTEGER,             -- FK to active_tsl_orders.id or strategy_groups.id
    symbol           TEXT NOT NULL,
    exchange         TEXT NOT NULL,
    transaction_type TEXT NOT NULL,
    quantity         INTEGER NOT NULL,
    order_type       TEXT NOT NULL,       -- 'MARKET' | 'LIMIT' | 'GTT'
    price            REAL,
    trigger_price    REAL,
    order_id         TEXT,                -- Kite order_id or gtt trigger_id returned
    status           TEXT DEFAULT 'placed',  -- 'placed' | 'failed' | 'executed'
    error_message    TEXT,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Open (or create) the trading DB and ensure all required tables exist."""
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.executescript(_SCHEMA_SQL)
    db.commit()
    return db


# ─────────────────────────────────────────────
#  Order Logging Helper
# ─────────────────────────────────────────────

def log_order(db, *, source, source_id=None, symbol, exchange,
              transaction_type, quantity, order_type, price=None,
              trigger_price=None, order_id=None, status="placed",
              error_message=None):
    """Insert a row into order_log.  Every order placed should go through here."""
    cur = db.cursor()
    cur.execute("""
        INSERT INTO order_log
            (source, source_id, symbol, exchange, transaction_type,
             quantity, order_type, price, trigger_price, order_id,
             status, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (source, source_id, symbol, exchange, transaction_type,
          quantity, order_type, price, trigger_price,
          str(order_id) if order_id else None, status, error_message))
    db.commit()
    return cur.lastrowid


# ─────────────────────────────────────────────
#  Helper: create_group()
# ─────────────────────────────────────────────

def create_group(db, name, trigger_mode, trailing_percent, *,
                 ref_symbol=None, ref_exchange=None,
                 legs=None):
    """
    Create a strategy group and optionally insert its legs in one call.

    Parameters
    ----------
    db              : sqlite3.Connection
    name            : str — human-readable name, e.g. "GOLDM Synthetic Put Mar26"
    trigger_mode    : str — 'combined_pnl' or 'underlying'
    trailing_percent: float
    ref_symbol      : str — required when trigger_mode == 'underlying'
    ref_exchange    : str — required when trigger_mode == 'underlying'
    legs            : list[dict] — each dict has keys:
                        symbol, exchange, quantity, transaction_type,
                        entry_price, product (optional, default 'NRML')

    Returns
    -------
    int — newly created group_id
    """
    if trigger_mode not in ("combined_pnl", "underlying"):
        raise ValueError(f"Invalid trigger_mode: {trigger_mode!r}. "
                         f"Must be 'combined_pnl' or 'underlying'.")

    if trigger_mode == "underlying" and (not ref_symbol or not ref_exchange):
        raise ValueError("ref_symbol and ref_exchange are required "
                         "when trigger_mode == 'underlying'.")

    cur = db.cursor()
    cur.execute("""
        INSERT INTO strategy_groups
            (name, trigger_mode, ref_symbol, ref_exchange,
             trailing_percent, hwm_value, status)
        VALUES (?, ?, ?, ?, ?, 0, 'active')
    """, (name, trigger_mode, ref_symbol, ref_exchange, trailing_percent))
    group_id = cur.lastrowid
    db.commit()

    print(f"✅ Group created: [{group_id}] {name!r}  "
          f"mode={trigger_mode}  trail={trailing_percent}%")

    if legs:
        for leg in legs:
            add_leg(db, group_id, **leg)

    return group_id


# ─────────────────────────────────────────────
#  Helper: add_leg / remove_leg
# ─────────────────────────────────────────────

def add_leg(db, group_id, symbol, exchange, quantity, transaction_type,
            entry_price, product="NRML"):
    """
    Add a leg to a strategy group.
    Also removes any matching individual TSL from active_tsl_orders.
    """
    cur = db.cursor()

    # Insert the leg
    cur.execute("""
        INSERT INTO strategy_group_legs
            (group_id, symbol, exchange, quantity, transaction_type,
             entry_price, product, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
    """, (group_id, symbol, exchange, quantity,
          transaction_type, entry_price, product))
    leg_id = cur.lastrowid

    # Remove matching individual TSL entry (if any)
    cur.execute("""
        UPDATE active_tsl_orders
        SET status = 'moved_to_group', updated_at = CURRENT_TIMESTAMP
        WHERE symbol = ? AND exchange = ? AND status = 'active'
    """, (symbol, exchange))
    moved = cur.rowcount

    db.commit()

    flag = f"  (removed {moved} individual TSL)" if moved else ""
    print(f"  📌 Leg added [{leg_id}]: {transaction_type} {quantity}x "
          f"{exchange}:{symbol} @ {entry_price}{flag}")
    return leg_id


def remove_leg(db, leg_id):
    """Soft-close a single leg (sets status = 'closed')."""
    cur = db.cursor()
    cur.execute("""
        UPDATE strategy_group_legs
        SET status = 'closed'
        WHERE id = ?
    """, (leg_id,))
    db.commit()
    print(f"  ❌ Leg [{leg_id}] closed.")


# ─────────────────────────────────────────────
#  StrategyGroup Class
# ─────────────────────────────────────────────

class StrategyGroup:
    """
    Manages a multi-leg strategy as a single unit for trailing stop-loss.

    Usage:
        group = StrategyGroup(kite, db, group_id)
        group.register_with_monitor(price_monitor)
        group.update_trailing_sl(price_monitor)   # call each tick
    """

    def __init__(self, kite, db, group_id):
        self.kite = kite
        self.db = db
        self.group_id = group_id
        self.meta = None
        self.legs = []
        self._load()

    # ── DB I/O ──

    def _load(self):
        """Load group metadata and active legs from DB."""
        cur = self.db.cursor()

        cur.execute("SELECT * FROM strategy_groups WHERE id = ?",
                    (self.group_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Group {self.group_id} not found.")

        self.meta = dict(row)

        cur.execute("""
            SELECT * FROM strategy_group_legs
            WHERE group_id = ? AND status = 'active'
        """, (self.group_id,))
        self.legs = [dict(r) for r in cur.fetchall()]

    def _save_meta(self):
        """Persist HWM / trigger updates to DB."""
        cur = self.db.cursor()
        cur.execute("""
            UPDATE strategy_groups
            SET hwm_value = ?, last_trigger_value = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (self.meta["hwm_value"], self.meta["last_trigger_value"],
              self.group_id))
        self.db.commit()

    # ── Price / P&L helpers ──

    def compute_combined_pnl(self, price_monitor):
        """
        Sum of unrealized P&L across all active legs.

        For a BUY leg:  pnl = (current_price - entry_price) * qty
        For a SELL leg: pnl = (entry_price - current_price) * qty
        """
        total_pnl = 0.0
        for leg in self.legs:
            price = price_monitor.get_price(leg["exchange"], leg["symbol"])
            if price is None:
                print(f"  ⚠️ No price for {leg['symbol']}, skipping leg")
                continue

            entry = leg["entry_price"] or 0
            qty = leg["quantity"]

            if leg["transaction_type"] == "BUY":
                pnl = (price - entry) * qty
            else:  # SELL
                pnl = (entry - price) * qty

            total_pnl += pnl

        return total_pnl

    def get_trigger_value(self, price_monitor):
        """
        Return the numeric value that drives the trailing SL.
        Depends on trigger_mode.
        """
        mode = self.meta["trigger_mode"]

        if mode == "combined_pnl":
            return self.compute_combined_pnl(price_monitor)

        elif mode == "underlying":
            ref_price = price_monitor.get_price(
                self.meta["ref_exchange"], self.meta["ref_symbol"])
            if ref_price is None:
                print(f"  ⚠️ No price for underlying "
                      f"{self.meta['ref_exchange']}:{self.meta['ref_symbol']}")
            return ref_price

        else:
            raise ValueError(f"Unknown trigger_mode: {mode!r}")

    # ── Monitor integration ──

    def register_with_monitor(self, price_monitor):
        """Ensure all leg symbols (+ ref underlying) are being tracked."""
        for leg in self.legs:
            price_monitor.add_symbol(leg["exchange"], leg["symbol"])

        if self.meta["trigger_mode"] == "underlying":
            price_monitor.add_symbol(
                self.meta["ref_exchange"], self.meta["ref_symbol"])

    # ── Core trailing SL logic ──

    def update_trailing_sl(self, price_monitor):
        """
        One-tick update of the group's trailing stop-loss.

        1. Get current trigger value (combined P&L or underlying price).
        2. Update HWM if new high.
        3. Calculate SL threshold.
        4. If current value has dropped through SL → fire exit for all legs.

        Returns True if SL was triggered, False otherwise.
        """
        trigger_val = self.get_trigger_value(price_monitor)
        if trigger_val is None:
            return False

        mode = self.meta["trigger_mode"]
        hwm = self.meta["hwm_value"]
        trail_pct = self.meta["trailing_percent"]

        # --- Update HWM ---
        if trigger_val > hwm:
            old_hwm = hwm
            self.meta["hwm_value"] = trigger_val
            hwm = trigger_val
            self._save_meta()
            label = "P&L" if mode == "combined_pnl" else "Price"
            print(f"  🚀 [{self.meta['name']}] New HWM! "
                  f"{label}: {old_hwm:.2f} → {trigger_val:.2f}")

        # --- Calculate SL threshold ---
        if mode == "combined_pnl":
            # For P&L: SL fires when P&L drops below HWM - (HWM * trail%)
            # But if HWM is negative (losing position), no trailing applies
            if hwm <= 0:
                sl_threshold = hwm  # don't trail negative HWM
            else:
                sl_threshold = hwm * (1 - trail_pct / 100)
        else:
            # For underlying price: SL fires when price drops X% from HWM
            sl_threshold = hwm * (1 - trail_pct / 100)

        sl_threshold = round(sl_threshold * 20) / 20  # tick-round

        # --- Update stored trigger if it moved up ---
        old_trigger = self.meta["last_trigger_value"]
        if sl_threshold > old_trigger:
            self.meta["last_trigger_value"] = sl_threshold
            self._save_meta()
            print(f"  ⚡ [{self.meta['name']}] SL trailed up: "
                  f"{old_trigger:.2f} → {sl_threshold:.2f}")

        # --- Check if SL is breached ---
        if trigger_val <= self.meta["last_trigger_value"] and self.meta["last_trigger_value"] > 0:
            label = "P&L" if mode == "combined_pnl" else "Price"
            print(f"  🔴 [{self.meta['name']}] SL TRIGGERED!  "
                  f"{label}={trigger_val:.2f} ≤ SL={self.meta['last_trigger_value']:.2f}")
            self.place_exit_orders()
            return True

        # --- Status printout ---
        label = "P&L" if mode == "combined_pnl" else "Price"
        print(f"  🎯 [{self.meta['name']}] {label}={trigger_val:.2f}  "
              f"HWM={hwm:.2f}  SL@{trail_pct}%={self.meta['last_trigger_value']:.2f}")
        return False

    # ── Exit logic ──

    def place_exit_orders(self):
        """
        Market-exit every active leg in the group.
        Logs each order to order_log.
        """
        print(f"\n  🚨 Exiting all legs for group [{self.group_id}] "
              f"{self.meta['name']!r}")

        for leg in self.legs:
            # Reverse the transaction type for exit
            exit_txn = ("SELL" if leg["transaction_type"] == "BUY"
                        else "BUY")
            sym = leg["symbol"]
            exch = leg["exchange"]
            qty = leg["quantity"]
            product = leg["product"] or "NRML"

            try:
                order_id = self.kite.place_order(
                    variety=self.kite.VARIETY_REGULAR,
                    exchange=exch,
                    tradingsymbol=sym,
                    transaction_type=exit_txn,
                    quantity=qty,
                    product=product,
                    order_type=self.kite.ORDER_TYPE_MARKET,
                )
                print(f"    ✅ EXIT {exit_txn} {qty}x {sym}  "
                      f"order_id={order_id}")

                log_order(self.db,
                          source="group_exit", source_id=self.group_id,
                          symbol=sym, exchange=exch,
                          transaction_type=exit_txn, quantity=qty,
                          order_type="MARKET", order_id=order_id,
                          status="placed")

            except Exception as e:
                print(f"    ❌ EXIT FAILED {sym}: {e}")

                log_order(self.db,
                          source="group_exit", source_id=self.group_id,
                          symbol=sym, exchange=exch,
                          transaction_type=exit_txn, quantity=qty,
                          order_type="MARKET", status="failed",
                          error_message=str(e))

        # Mark group and legs as closed
        cur = self.db.cursor()
        cur.execute("""
            UPDATE strategy_groups
            SET status = 'closed', updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (self.group_id,))
        cur.execute("""
            UPDATE strategy_group_legs
            SET status = 'closed'
            WHERE group_id = ? AND status = 'active'
        """, (self.group_id,))
        self.db.commit()

        # Reload to reflect changes
        self._load()

    # ── Display ──

    def __repr__(self):
        n_legs = len(self.legs)
        return (f"<StrategyGroup [{self.group_id}] {self.meta['name']!r}  "
                f"mode={self.meta['trigger_mode']}  legs={n_legs}  "
                f"status={self.meta['status']}>")

    def summary(self):
        """Pretty-print group status."""
        m = self.meta
        print(f"\n{'='*60}")
        print(f"📦 Group [{m['id']}]: {m['name']}")
        print(f"   Mode: {m['trigger_mode']}  |  Trail: {m['trailing_percent']}%  |  Status: {m['status']}")
        if m["trigger_mode"] == "underlying":
            print(f"   Underlying: {m['ref_exchange']}:{m['ref_symbol']}")
        print(f"   HWM: {m['hwm_value']:.2f}  |  SL: {m['last_trigger_value']:.2f}")
        print(f"   Legs:")
        for leg in self.legs:
            print(f"     [{leg['id']}] {leg['transaction_type']} {leg['quantity']}x "
                  f"{leg['exchange']}:{leg['symbol']} @ {leg['entry_price']}")
        print(f"{'='*60}")


# ─────────────────────────────────────────────
#  Convenience: load all active groups
# ─────────────────────────────────────────────

def load_active_groups(kite, db):
    """Return a list of StrategyGroup objects for every active group."""
    cur = db.cursor()
    cur.execute("SELECT id FROM strategy_groups WHERE status = 'active'")
    return [StrategyGroup(kite, db, row["id"]) for row in cur.fetchall()]


# ─────────────────────────────────────────────
#  Convenience: view order log
# ─────────────────────────────────────────────

def get_order_log(db, limit=50, source=None):
    """Fetch recent entries from order_log."""
    cur = db.cursor()
    if source:
        cur.execute("""
            SELECT * FROM order_log
            WHERE source = ?
            ORDER BY id DESC LIMIT ?
        """, (source, limit))
    else:
        cur.execute("""
            SELECT * FROM order_log
            ORDER BY id DESC LIMIT ?
        """, (limit,))
    return [dict(r) for r in cur.fetchall()]


def print_order_log(db, limit=20, source=None):
    """Pretty-print recent orders."""
    rows = get_order_log(db, limit=limit, source=source)
    if not rows:
        print("No orders in log.")
        return

    print(f"\n{'Source':<14} {'Symbol':<28} {'Type':<6} {'Qty':>4} "
          f"{'OrdType':<7} {'Price':>10} {'Status':<8} {'Time'}")
    print("-" * 105)
    for r in rows:
        price_str = f"{r['price']:.2f}" if r["price"] else "-"
        print(f"{r['source']:<14} {r['symbol']:<28} {r['transaction_type']:<6} "
              f"{r['quantity']:>4} {r['order_type']:<7} {price_str:>10} "
              f"{r['status']:<8} {r['created_at']}")
