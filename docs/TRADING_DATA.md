# Trading Proxy Data

**DISCLAIMER**: This is a real-market proxy dataset created because the assessment did not provide historical trading logs or strategy parameter definitions. It is NOT official CrowdWisdomTrading data.

## Data Source
- **Provider**: `yfinance`
- **Instrument**: SPY (SPDR S&P 500 ETF Trust)
- **Timeframe**: Daily OHLCV (2000 - Present)

## Trading Strategy
- **Type**: Simple Moving Average (SMA) Crossover
- **Parameters**: 
  - SMA 5/20
  - SMA 10/30
  - SMA 20/50
  - SMA 50/100

## Execution Rules & Leakage Prevention
1. **Signal Generation**: A crossover is evaluated using the Close price at the end of Day T.
2. **Execution Timing**: If a crossover occurs on Day T, the trade is executed at the Open price on Day T+1. This strict offset prevents look-ahead bias (data leakage).
3. **Positioning**: 
   - Long (+1) when Fast SMA > Slow SMA
   - Short (-1) when Fast SMA < Slow SMA
4. **Exit**: An open position is exited at the Open of the day following a reverse crossover signal.

## P&L Calculation
- **Long PNL**: `exit_price - entry_price`
- **Short PNL**: `entry_price - exit_price`
- **Return %**: `PNL / entry_price`
- **Win Flag**: 1 if PNL > 0 else 0

Transaction costs and slippage are not modeled in this MVP to preserve a pure target for the ML algorithms.
