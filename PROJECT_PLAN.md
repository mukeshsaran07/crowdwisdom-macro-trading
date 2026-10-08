# Project Plan: CrowdWisdomTrading Macro-Aware Trading ML MVP

## Objective

Build an MVP to evaluate trading strategies using market data, macroeconomic events, economic news, and machine learning.

## Data Sources

- **Market Data:** SPY daily data used as a transparent proxy because official trading logs were unavailable.
- **Strategies:** SMA 5/20, 10/30, 20/50, and 50/100.
- **Macro Data:** U.S. economic events from Apify.
- **News:** Recent macroeconomic news from Tavily.

## Pipeline

```text
Market Data + Macro Events + News
              ↓
      Feature Engineering
              ↓
       ML Walk-Forward
          Validation
              ↓
       Trade Selection
              ↓
     Strategy Evaluation
              ↓
        Final Report
