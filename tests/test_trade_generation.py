import pandas as pd
import numpy as np
import pytest
from src.data.trade_generator import generate_trades

def test_trade_generation():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    
    closes = [10, 10, 10, 10, 20, 20, 20, 5, 5, 5]
    opens = [10, 10, 10, 10, 15, 20, 20, 15, 5, 5]
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': closes,
        'low': closes,
        'close': closes,
        'volume': 100
    })
    
    trades = generate_trades(df, fast_window=2, slow_window=4, symbol="TEST")
    
    assert len(trades) > 0, "No trades were generated."
    
    first_trade = trades[0]
    
    # Executed at Day 5 open (T=5) which is '2023-01-06'
    assert first_trade['position'] == 1, "First closed trade should be a long exit."
    assert first_trade['entry_price'] == 20.0
    assert first_trade['exit_price'] == 5.0
    assert first_trade['pnl'] == (5.0 - 20.0)
    assert first_trade['win_flag'] == 0
    
    # Leakage check: Bullish cross on Day 4 (T=4), executed on Day 5 (T=5)
    assert first_trade['entry_time'] == pd.Timestamp("2023-01-06")

    # Second strategy permutation check
    trades2 = generate_trades(df, fast_window=3, slow_window=5, symbol="TEST")
    assert isinstance(trades2, list)
