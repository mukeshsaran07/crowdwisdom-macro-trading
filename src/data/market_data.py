import yfinance as yf
import pandas as pd

def fetch_market_data(symbol: str, start_date: str = "2000-01-01", end_date: str = None) -> pd.DataFrame:
    """
    Fetches historical OHLCV data from yfinance.
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    # yfinance returns MultiIndex columns sometimes in newer versions, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    df = df.reset_index()
    # Ensure columns are lowercase
    df.columns = [col.lower() for col in df.columns]
    # Rename Date to timestamp
    if 'date' in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    elif 'datetime' in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    
    df['symbol'] = symbol
    df['data_source'] = 'yfinance'
    
    # Ensure proper types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'data_source']]
