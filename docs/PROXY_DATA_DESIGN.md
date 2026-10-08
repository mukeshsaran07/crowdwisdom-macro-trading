# Proxy Data Design

## 1. Why a proxy is required
No official historical trading logs or strategy parameter specifications were provided in the CrowdWisdomTrading assessment materials or public repositories. To complete the assessment within the 3-day timeline while strictly avoiding fabricated or fake data, we must generate a transparent, reproducible set of real trades based on actual market data. This approach demonstrates the ability to build an end-to-end quantitative ML pipeline without compromising data integrity.

## 2. Market instrument
**SPY (SPDR S&P 500 ETF Trust)**
*Reasoning*: SPY is highly liquid, has decades of reliable daily OHLCV data accessible via `yfinance`, and is the most representative instrument of the broader US economy. This makes it highly sensitive and relevant to the macroeconomic event data and economic news we will scrape.

## 3. Trading strategy
**Simple Moving Average (SMA) Crossover**
*Reasoning*: The SMA crossover is easy to explain, computationally inexpensive, and widely understood. It provides a clear, rule-based mechanism to generate entries and exits without requiring complex optimizations, fitting perfectly within an MVP scope.

## 4. Strategy permutations
We will evaluate the following 4 parameter permutations (Fast SMA / Slow SMA):
1. **10 / 30**: Short-term responsive trend tracking.
2. **20 / 50**: Medium-term standard trend following.
3. **50 / 100**: Long-term macro trend following.
4. **5 / 20**: Very short-term momentum.
*Reasoning*: These are standard, sensible window sizes that capture different market regimes (short, medium, and long-term trends) without overfitting to specific noise.

## 5. Trade-generation rules
- **Entry Condition**: A crossover event occurs at the end of Day *T*. (e.g., Fast SMA > Slow SMA for a Long signal).
- **Execution (Entry Timestamp)**: The trade is executed at the Open price of Day *T+1*. (Crucial to prevent look-ahead bias).
- **Exit Condition**: The reverse crossover event occurs at the end of Day *X*. The position is exited at the Open price of Day *X+1*.
- **Position Direction**: Continuous Long/Short. 
  - `+1` (Long) when Fast SMA crosses above Slow SMA.
  - `-1` (Short) when Fast SMA crosses below Slow SMA.
- **P&L Calculation**: `(Exit Price - Entry Price) * Position Direction`.
- **Transaction Costs**: Ignored for the MVP to keep the target variable simple, but we will explicitly state this limitation in the final report.

## 6. Trading-log schema
The generated trades will be stored in SQLite with the following schema:
- `trade_id` (String/UUID): Unique identifier for the trade.
- `strategy_id` (String): e.g., "SMA_10_30".
- `fast_window` (Integer): The fast SMA parameter.
- `slow_window` (Integer): The slow SMA parameter.
- `entry_time` (Datetime): Timestamp of trade entry (Day T+1 Open).
- `exit_time` (Datetime): Timestamp of trade exit (Day X+1 Open).
- `entry_price` (Float): Asset price at entry.
- `exit_price` (Float): Asset price at exit.
- `position` (Integer): +1 for Long, -1 for Short.
- `pnl` (Float): Absolute profit/loss.
- `return_pct` (Float): Percentage return on the trade.
- `win_flag` (Integer): 1 if pnl > 0, else 0.

## 7. Macro-data joining strategy
To predict trade outcome (P&L or Win Rate), macroeconomic features must be joined to each trade at the time of entry. We will use an "as-of" join: for a trade with `entry_time` $T$, we will aggregate or select macroeconomic events and news sentiment that occurred in the window $[T - \Delta, T)$. This ensures the model only sees macro data that was publicly available strictly before the trade was executed.

## 8. Leakage prevention
- **Price Data**: Trade entries/exits occur at the *Open* of the day following a signal, preventing the model from acting on the closing price of the signal day.
- **Macro Data**: All macro data timestamps will be strictly restricted to `< entry_time`.
- **Feature Engineering**: Features (e.g., moving averages) will be calculated prior to splitting the dataset chronologically.

## 9. Walk-forward validation design
We will evaluate the ML model using chronological walk-forward validation (TimeSeriesSplit).
- The dataset will be ordered by `entry_time`.
- We will use expanding or rolling windows (e.g., Train on months 1-3, Test on month 4; Train on months 1-4, Test on month 5).
- Random `train_test_split` is strictly prohibited to maintain the time-series integrity and avoid look-ahead bias.

## 10. Limitations
- **Proxy Data**: These are simulated trades, not actual historical performance logs of a proprietary fund strategy.
- **Simplistic Strategy**: The SMA crossover is a baseline strategy and is unlikely to be highly profitable without transaction costs and slippage modeling.
- **Missing Frictions**: Does not account for slippage, broker commissions, or liquidity constraints.

## 11. Reproducibility
The exact code used to download `yfinance` data and generate the trades will be included in the repository. A random seed will be set where applicable, and the `requirements.txt` will freeze package versions.
