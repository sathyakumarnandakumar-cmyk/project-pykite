
#class RuleStatus(str, enum.Enum):
#    PENDING = "PENDING"
#    FILLED = "FILLED"
#    CANCELLED = "CANCELLED"



from sqlalchemy import Column, Integer, String, Float, ForeignKey, Enum, DateTime, UUID
from sqlalchemy.orm import relationship, declarative_base
import enum
import uuid
import datetime

Base = declarative_base()

class RuleType(enum.Enum):
    ABSOLUTE_SL = "ABSOLUTE_SL"
    TRAILING_SL = "TRAILING_SL"
    ABSOLUTE_TARGET = "ABSOLUTE_TARGET"
    PARTIAL_TARGET = "PARTIAL_TARGET"

class RuleStatus(enum.Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"

class Strategy(Base):
    __tablename__ = "strategies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String, nullable=False)
    total_quantity = Column(Integer, nullable=False)  # Current shares managed
    status = Column(String, default="ACTIVE")        # ACTIVE, COMPLETED
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship to individual triggers
    rules = relationship("OrderRule", back_populates="strategy", cascade="all, delete-orphan")

class OrderRule(Base):
    __tablename__ = "order_rules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey("strategies.id"))
    
    rule_type = Column(Enum(RuleType), nullable=False)
    status = Column(Enum(RuleStatus), default=RuleStatus.PENDING)
    
    # Price Logic
    trigger_price = Column(Float, nullable=True)     # The price to fire at
    quantity_percent = Column(Float, nullable=False) # 0.1 to 1.0
    
    # Trailing Logic
    trailing_percent = Column(Float, nullable=True)  # e.g., 0.05 for 5%
    high_water_mark = Column(Float, nullable=True)   # The peak price seen
    
    strategy = relationship("Strategy", back_populates="rules")







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