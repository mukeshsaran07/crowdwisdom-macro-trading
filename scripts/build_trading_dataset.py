import sys
import os
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.market_data import fetch_market_data
from src.data.trade_generator import generate_trades
from src.data.database import init_db, MarketPrice, StrategyTrade

def main():
    print("Initializing database...")
    db_path = "sqlite:///data/macro_trading.db"
    Session = init_db(db_path)
    session = Session()

    print("Fetching SPY historical data from yfinance...")
    symbol = "SPY"
    try:
        df_market = fetch_market_data(symbol, start_date="2000-01-01")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    print(f"Downloaded {len(df_market)} rows of market data.")
    print(f"Date range: {df_market['timestamp'].min().date()} to {df_market['timestamp'].max().date()}")

    print("Saving market prices to database...")
    session.query(MarketPrice).delete()
    session.commit()
    df_market.to_sql('market_prices', session.get_bind(), if_exists='append', index=False)
    
    permutations = [
        (5, 20),
        (10, 30),
        (20, 50),
        (50, 100)
    ]
    
    all_trades = []
    
    print("Generating trades for strategy permutations...")
    for fast, slow in permutations:
        trades = generate_trades(df_market, fast, slow, symbol)
        all_trades.extend(trades)
        
    df_trades = pd.DataFrame(all_trades)
    
    if not df_trades.empty:
        print("Saving trades to database...")
        session.query(StrategyTrade).delete()
        session.commit()
        # Convert datetime columns properly if needed
        df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
        df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
        df_trades.to_sql('strategy_trades', session.get_bind(), if_exists='append', index=False)
            
        csv_path = Path("data/processed/strategy_trades.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_trades.to_csv(csv_path, index=False)
        print(f"Saved CSV copy to {csv_path}")
        
        print("\n--- Strategy Performance Summary ---")
        summary = df_trades.groupby('strategy_id').agg(
            total_trades=('trade_id', 'count'),
            total_pnl=('pnl', 'sum'),
            win_rate=('win_flag', lambda x: x.mean())
        ).reset_index()
        
        print(summary.to_string(index=False))
    else:
        print("No trades generated.")
        
    print("\nProcess completed successfully.")

if __name__ == "__main__":
    main()
