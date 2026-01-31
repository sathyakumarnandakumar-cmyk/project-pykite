#class RuleStatus(str, enum.Enum):
#    PENDING = "PENDING"
#    FILLED = "FILLED"
#    CANCELLED = "CANCELLED"


async def process_strategy(strategy: Strategy, current_price: Decimal):
    for rule in strategy.rules:
        if rule.status != RuleStatus.PENDING:
            continue

        # 1. Logic for Trailing Stop Loss (Update High Water Mark)
        if rule.type == RuleType.TRAILING_SL:
            if rule.high_water_mark is None or current_price > rule.high_water_mark:
                rule.high_water_mark = current_price
                # Recalculate the actual trigger price based on the peak
                rule.trigger_price = rule.high_water_mark * (1 - rule.trailing_percent)
                # Note: Save this to DB in a real app

        # 2. Check Triggers
        is_triggered = False
        if "target" in rule.type and current_price >= rule.trigger_price:
            is_triggered = True
        elif "stop_loss" in rule.type and current_price <= rule.trigger_price:
            is_triggered = True

        # 3. Handle Execution and OCO Logic
        if is_triggered:
            await execute_trade(strategy, rule)
            
            # Update Rule Status
            rule.status = RuleStatus.FILLED
            
            # handle the Side-Effects
            if rule.quantity_percent == 1.0:
                # ABSOLUTE: Cancel everything else
                for other_rule in strategy.rules:
                    if other_rule.id != rule.id:
                        other_rule.status = RuleStatus.CANCELLED
                strategy.status = "completed"
                break 
            else:
                # PARTIAL: Adjust remaining quantity for other rules
                reduction_factor = (1 - rule.quantity_percent)
                strategy.current_quantity = int(strategy.current_quantity * reduction_factor)
                
                # If remaining qty is negligible, close strategy
                if strategy.current_quantity <= 0:
                    strategy.status = "completed"