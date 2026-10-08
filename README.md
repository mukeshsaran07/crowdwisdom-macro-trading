# CrowdWisdomTrading Macro Trading MVP

## Project Overview
This project builds a quantitative ML pipeline to predict trade outcomes based on macroeconomic data and strategy permutations for the CrowdWisdomTrading assessment MVP.

## Data Pipeline
**Disclaimer:** As the assessment did not provide official trading logs, we utilize a transparent real-data proxy.
- Historical prices are sourced from `yfinance` (SPY ETF).
- Trades are generated using SMA crossovers.
- Strict T+1 execution rules are applied to prevent look-ahead bias.
- Trades and prices are stored in an SQLite database.

See `docs/TRADING_DATA.md` for full details.

## Macro Data Pipeline
Macroeconomic events (last 180 days) are sourced using the Apify API (`pintostudio/economic-calendar-data-investing-com`).
1. Set `APIFY_API_TOKEN` in `.env`.
2. Run `python scripts/fetch_macro_data.py` (use `--mock` if no token is available).
See `docs/MACRO_DATA.md` for full details.
