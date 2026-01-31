# More conditioned. Avoids double selling at all cases and more atomic in nature.
# We add a broker_order_id to track the live order on the exchange and a new status SUBMITTED.
# StatusMeaningPENDINGMonitoring price; nothing sent to broker yet.SUBMITTEDOrder sent to broker; waiting for fill (Limit order).FILLEDOrder successfully executed.CANCELLEDRule invalidated or manually stopped.
# 

# The main challenge is that an order can be "Pending" in your database (waiting for a price trigger) OR "Pending" at the Broker/Exchange (waiting for a buyer). If a second, more urgent rule (like an Absolute Stop Loss) triggers while a previous partial order is still sitting on the exchange, the system must explicitly send a cancel request to the broker for the first order before (or while) sending the new one.

# Why this solves the "5-minute delay" problem:
# Broker Sync: Every minute, the system first asks the broker, "Did that limit order from 5 minutes ago fill?" If no, it stays SUBMITTED.

# The Panic Button (Absolute SL): If the market crashes and the Absolute SL triggers, the code looks for any rule marked SUBMITTED. It calls broker.cancel_order() to pull that 5-minute-old limit order off the books.

# Clean State: By cancelling the old order first, you prevent a "Double Sell" (where the limit order fills and the market order fills, leaving you short).


import asyncio
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession

class RuleStatus(str, Enum):
    PENDING = "PENDING"      # Waiting for price trigger
    SUBMITTED = "SUBMITTED"  # Live on exchange (Limit order)
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"

async def process_strategy(session: AsyncSession, strategy: Strategy, current_price: float, broker):
    """
    Handles complex state: If an Absolute rule triggers, 
    it must cancel any existing 'SUBMITTED' broker orders.
    """
    
    # 1. First, check if any 'SUBMITTED' orders were filled since the last minute
    for rule in strategy.rules:
        if rule.status == RuleStatus.SUBMITTED:
            # Sync with Broker API
            is_filled = await broker.get_order_status(rule.broker_order_id)
            if is_filled:
                rule.status = RuleStatus.FILLED
                strategy.total_quantity -= rule.last_calculated_qty 
                # If quantity hits 0, close strategy
                if strategy.total_quantity <= 0:
                    strategy.status = "COMPLETED"
                    return # Exit early

    # 2. Check Triggers for PENDING rules
    for rule in strategy.rules:
        if rule.status != RuleStatus.PENDING:
            continue
        
        # (Trailing logic update happens here as before...)
        
        if is_triggered(rule, current_price):
            # We found a trigger! (e.g., Absolute Stop Loss)
            
            # --- THE "SWEEP" LOGIC ---
            # If this is an Absolute SL (qty_percent=1.0), 
            # we MUST cancel any other SUBMITTED orders at the broker first.
            if rule.quantity_percent == 1.0:
                for other_rule in strategy.rules:
                    if other_rule.status == RuleStatus.SUBMITTED:
                        await broker.cancel_order(other_rule.broker_order_id)
                        other_rule.status = RuleStatus.CANCELLED
            
            # 3. Execute the new triggered order
            shares_to_sell = int(strategy.total_quantity * rule.quantity_percent)
            
            # Send to broker (assuming a Market or Limit order)
            order_resp = await broker.send_order(
                ticker=strategy.ticker, 
                qty=shares_to_sell, 
                order_type="MARKET" if rule.quantity_percent == 1.0 else "LIMIT"
            )
            
            # 4. Update Database State
            rule.broker_order_id = order_resp['id']
            rule.status = RuleStatus.SUBMITTED # Or FILLED immediately if Market order
            
            # If Absolute, mark all other PENDING rules as CANCELLED immediately
            if rule.quantity_percent == 1.0:
                for r in strategy.rules:
                    if r.id != rule.id and r.status == RuleStatus.PENDING:
                        r.status = RuleStatus.CANCELLED
            
            # Commit changes for this strategy
            await session.commit()
            break # Avoid processing multiple triggers in one tick