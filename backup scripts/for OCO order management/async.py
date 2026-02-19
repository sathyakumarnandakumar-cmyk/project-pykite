from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import update, select

async def process_order_logic(session: AsyncSession, strategy: Strategy, current_price: float):
    """
    Evaluates rules for a single strategy and handles OCO/Partial logic.
    """
    for rule in strategy.rules:
        if rule.status != RuleStatus.PENDING:
            continue

        # 1. Update Trailing Stop Logic (Stateful)
        if rule.rule_type == RuleType.TRAILING_SL:
            if rule.high_water_mark is None or current_price > rule.high_water_mark:
                rule.high_water_mark = current_price
                rule.trigger_price = rule.high_water_mark * (1 - rule.trailing_percent)
                # We update the DB object; SQLAlchemy tracks changes for the commit

        # 2. Check if Triggered
        is_triggered = False
        if "TARGET" in rule.rule_type.value and current_price >= rule.trigger_price:
            is_triggered = True
        elif "SL" in rule.rule_type.value and current_price <= rule.trigger_price:
            is_triggered = True

        if is_triggered:
            # Calculate actual shares to sell based on CURRENT strategy quantity
            shares_to_sell = int(strategy.total_quantity * rule.quantity_percent)
            
            # --- START EXECUTION & ATOMIC UPDATE ---
            # In a real app, call broker_api.send_order() here
            
            rule.status = RuleStatus.FILLED
            
            if rule.quantity_percent >= 1.0:
                # ABSOLUTE SELL: OCO (One Cancels Others)
                # Cancel all other pending rules for this strategy
                for other_rule in strategy.rules:
                    if other_rule.id != rule.id and other_rule.status == RuleStatus.PENDING:
                        other_rule.status = RuleStatus.CANCELLED
                
                strategy.total_quantity = 0
                strategy.status = "COMPLETED"
                break # Exit loop, strategy is done
            
            else:
                # PARTIAL SELL: Adjust strategy quantity
                strategy.total_quantity -= shares_to_sell
                
                if strategy.total_quantity <= 0:
                    strategy.status = "COMPLETED"
                    # Cancel remaining rules if quantity hits zero
                    for other_rule in strategy.rules:
                        if other_rule.status == RuleStatus.PENDING:
                            other_rule.status = RuleStatus.CANCELLED
                    break

    # Finalize all changes (updates trigger prices, statuses, and quantities)
    await session.commit()