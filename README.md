# CrowdWisdomTrading Macro-Aware Trading ML

A quantitative trading ML MVP that combines market data, macroeconomic events, and recent economic news to evaluate trading strategy performance using chronological walk-forward validation.

> **Important:** Official CrowdWisdomTrading historical trading logs and proprietary strategy definitions were not available in the supplied workspace. Therefore, SPY market data and transparent SMA crossover strategies were used as a proxy for this MVP.

## Overview

```text
Market Data + Macro Events + Economic News
                    ↓
           Feature Engineering
                    ↓
          ML Walk-Forward Validation
                    ↓
             Trade Selection
                    ↓
           Strategy Evaluation
                    ↓
              Final Report
Data Sources
Market Data: SPY daily market data
Macro Events: Apify economic calendar data
Economic News: Tavily
Storage: SQLite
ML: Scikit-learn
Strategy Permutations

The following SMA crossover strategies are evaluated:

SMA 5/20
SMA 10/30
SMA 20/50
SMA 50/100

Signals are generated using information available at market close and executed at the next trading day's open to reduce look-ahead bias.

Features

The model uses:

Time-based features
Macro event counts
High-impact event counts
Macro surprises
CPI, FOMC and employment events
Latest macro surprise
Days since previous macro event
Strategy parameters
Position

Only information available before trade entry is used.

Machine Learning

Chronological walk-forward validation is used instead of a random train/test split.

The ML model predicts trade outcomes and acts as a trade-selection filter.

Evaluation Metrics

Strategies are evaluated using:

P&L
Win Rate
Sharpe Ratio
Sortino Ratio
Maximum Drawdown
Profit Factor
Selection Rate
Macro Sentiment

Recent economic news is grouped into:

Employment
Monetary Policy
Financial Markets
Growth
Inflation

A simple keyword-based sentiment approach is used for exploratory macro analysis.

Project Structure
crowdwisdom_macro_trading/
├── data/
│   └── processed/
├── docs/
├── notebooks/
├── scripts/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── scraper/
├── tests/
├── .env.example
├── .gitignore
├── PROJECT_PLAN.md
├── README.md
└── requirements.txt
Important Outputs
data/processed/final_evaluation_report.md
data/processed/strategy_evaluation.csv
data/processed/walk_forward_report.csv
data/processed/walk_forward_predictions.csv
data/processed/model_dataset.csv
data/processed/macro_outlook.md
data/processed/macro_sentiment_matrix.csv
Installation
git clone https://github.com/mukeshsaran07/crowdwisdom-macro-trading.git
cd crowdwisdom-macro-trading

python -m venv .venv
# Activate the environment

pip install -r requirements.txt

Create a local .env file using .env.example and add the required API credentials if fresh external data retrieval is required.

Run
python scripts/build_trading_dataset.py
python scripts/build_macro_features.py
python scripts/train_walk_forward.py
python scripts/evaluate_strategies.py
python scripts/fetch_macro_news.py
python scripts/build_macro_sentiment.py
python scripts/build_macro_charts.py
python scripts/build_final_report.py

Run tests:

python -m pytest tests
Limitations

This is an MVP using transparent proxy data because official trading logs and proprietary strategy definitions were unavailable.

Transaction costs and slippage are not included. The macro sentiment model is exploratory, and historical performance metrics should not be interpreted as guaranteed future trading performance.

License

This project was created as part of a quantitative data science / machine learning assessment.


