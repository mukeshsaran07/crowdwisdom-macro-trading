import uuid
import pandas as pd

def generate_trades(df: pd.DataFrame, fast_window: int, slow_window: int, symbol: str) -> list:
    """
    Generates a list of trades based on SMA crossover logic.
    """
    df = df.sort_values('timestamp').copy()
    
    df['fast_sma'] = df['close'].rolling(window=fast_window).mean()
    df['slow_sma'] = df['close'].rolling(window=slow_window).mean()
    
    trades = []
    current_position = 0 # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    entry_time = None
    
    # Iterate through the DataFrame to simulate execution accurately
    for i in range(1, len(df) - 1):
        prev_fast = df['fast_sma'].iloc[i-1]
        prev_slow = df['slow_sma'].iloc[i-1]
        curr_fast = df['fast_sma'].iloc[i]
        curr_slow = df['slow_sma'].iloc[i]
        
        # Wait until we have enough data to calculate SMAs
        if pd.isna(curr_slow):
            continue
            
        bullish_cross = (curr_fast > curr_slow) and (prev_fast <= prev_slow)
        bearish_cross = (curr_fast < curr_slow) and (prev_fast >= prev_slow)
        
        # Trade is executed at the OPEN of the NEXT day (i+1)
        next_open = float(df['open'].iloc[i+1])
        next_time = df['timestamp'].iloc[i+1]
        
        if bullish_cross:
            if current_position == -1: # Exit short
                pnl = entry_price - next_open
                return_pct = pnl / entry_price if entry_price > 0 else 0
                win_flag = 1 if pnl > 0 else 0
                trades.append({
                    'trade_id': str(uuid.uuid4()),
                    'strategy_id': f'SMA_{fast_window}_{slow_window}',
                    'symbol': symbol,
                    'fast_window': fast_window,
                    'slow_window': slow_window,
                    'entry_time': entry_time,
                    'exit_time': next_time,
                    'entry_price': entry_price,
                    'exit_price': next_open,
                    'position': -1,
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'win_flag': win_flag,
                    'data_source': 'yfinance_proxy',
                    'proxy_flag': True
                })
            
            if current_position <= 0: # Enter long
                current_position = 1
                entry_price = next_open
                entry_time = next_time
                
        elif bearish_cross:
            if current_position == 1: # Exit long
                pnl = next_open - entry_price
                return_pct = pnl / entry_price if entry_price > 0 else 0
                win_flag = 1 if pnl > 0 else 0
                trades.append({
                    'trade_id': str(uuid.uuid4()),
                    'strategy_id': f'SMA_{fast_window}_{slow_window}',
                    'symbol': symbol,
                    'fast_window': fast_window,
                    'slow_window': slow_window,
                    'entry_time': entry_time,
                    'exit_time': next_time,
                    'entry_price': entry_price,
                    'exit_price': next_open,
                    'position': 1,
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'win_flag': win_flag,
                    'data_source': 'yfinance_proxy',
                    'proxy_flag': True
                })
                
            if current_position >= 0: # Enter short
                current_position = -1
                entry_price = next_open
                entry_time = next_time
                
    return trades
