import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class MarketPrice(Base):
    __tablename__ = 'market_prices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    symbol = Column(String, index=True)
    data_source = Column(String)

class StrategyTrade(Base):
    __tablename__ = 'strategy_trades'

    trade_id = Column(String, primary_key=True)
    strategy_id = Column(String)
    symbol = Column(String)
    fast_window = Column(Integer)
    slow_window = Column(Integer)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    entry_price = Column(Float)
    exit_price = Column(Float)
    position = Column(Integer)
    pnl = Column(Float)
    return_pct = Column(Float)
    win_flag = Column(Integer)
    data_source = Column(String)
    proxy_flag = Column(Boolean, default=True)

class MacroEvent(Base):
    __tablename__ = 'macro_events'

    event_id = Column(String, primary_key=True)
    event_time_utc = Column(DateTime, index=True)
    original_event_time = Column(String)
    original_timezone = Column(String)
    country = Column(String)
    currency = Column(String)
    event_name = Column(String)
    category = Column(String)
    importance = Column(String)
    actual_value = Column(Float)
    forecast_value = Column(Float)
    previous_value = Column(Float)
    macro_surprise = Column(Float)
    unit = Column(String)
    source = Column(String)
    scraped_at = Column(DateTime)

def get_engine(db_path="sqlite:///data/macro_trading.db"):
    db_file = db_path.replace("sqlite:///", "")
    os.makedirs(os.path.dirname(os.path.abspath(db_file)), exist_ok=True)
    return create_engine(db_path)

def init_db(db_path="sqlite:///data/macro_trading.db"):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
