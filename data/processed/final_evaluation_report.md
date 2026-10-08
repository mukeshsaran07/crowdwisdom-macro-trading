# CrowdWisdomTrading Macro Trading ML Evaluation

**Project type:** Macro-aware ML trading strategy evaluation

**Status:** Prototype / assessment implementation

> This report is a methodology demonstration using transparent proxy data where official CrowdWisdomTrading trading logs and strategy definitions were unavailable in the supplied workspace.

---

## 1. Strategy Evaluation

Four SMA crossover strategy permutations were evaluated using chronological walk-forward validation.

| Rank | Strategy | Test Trades | Baseline P&L | ML P&L | Baseline Win Rate | ML Win Rate | ML Sharpe | ML Sortino | ML Max DD | Profit Factor | Selection Rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | SMA_20_50 | 55 | 81.35 | 270.38 | 41.82% | 56.00% | 5.8427 | 17.1852 | -86.92 | 2.9127 | 45.45% |
| 2 | SMA_50_100 | 22 | 5.55 | 239.57 | 40.91% | 52.94% | 4.6703 | 17.3579 | -47.95 | 2.3937 | 77.27% |
| 3 | SMA_10_30 | 87 | 159.73 | 360.74 | 35.63% | 46.00% | 4.1564 | 15.7327 | -43.42 | 2.8158 | 57.47% |
| 4 | SMA_5_20 | 156 | 143.10 | 304.38 | 32.69% | 38.57% | 3.9034 | 13.3976 | -47.45 | 2.3566 | 44.87% |

## 2. Model-Selected Strategy

The evaluation ranking identifies **SMA_20_50** as the top-performing evaluated permutation under the ML-filtered walk-forward results.

- ML-filtered P&L: **270.38**
- ML-filtered win rate: **56.00%**
- Sharpe ratio: **5.8427**
- Sortino ratio: **17.1852**
- Maximum drawdown: **-86.92**
- Profit factor: **2.9127**
- Trade selection rate: **45.45%**

**Important:** These results are evaluation results on the project proxy dataset. They should not be interpreted as guaranteed future trading performance.

## 3. Walk-Forward Validation

The model uses chronological walk-forward evaluation rather than a random train/test split. This preserves the temporal ordering of market information.

| Strategy | Test Samples | MAE | RMSE | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SMA_10_30 | 87 | N/A | N/A | 56.32% | 42.86% | 67.74% | 52.50% |
| SMA_20_50 | 55 | N/A | N/A | 61.82% | 54.17% | 56.52% | 55.32% |
| SMA_50_100 | 22 | N/A | N/A | 54.55% | 44.44% | 44.44% | 44.44% |
| SMA_5_20 | 156 | N/A | N/A | 57.05% | 39.47% | 58.82% | 47.24% |

The model is therefore evaluated using information available up to each prediction point, reducing the risk of look-ahead leakage.

## 4. Macro News and Sentiment Analysis

Tavily returned **33 unique news articles** for the macro-news analysis. **20** articles fall within the most recent seven-day window.

Overall recent news sentiment was **Neutral (0.153)**.

| Theme | Recent 7d | Previous 7d | Change | Direction |
|---|---:|---:|---:|---|
| Employment | -0.074 | 0.200 | -0.274 | Deteriorating |
| Monetary Policy | 0.352 | 1.000 | -0.648 | Deteriorating |
| Financial Markets | 0.000 | N/A | N/A | Insufficient data |
| Growth | 0.209 | N/A | N/A | Insufficient data |
| Inflation | -1.000 | N/A | N/A | Insufficient data |

The sentiment score is a transparent, deterministic keyword-based indicator. It is used as an exploratory macro feature and should not be interpreted as a standalone financial forecast.

## 5. Data and Methodology

### Market Data

The project uses **SPY historical market data as a transparent
proxy dataset** because official CrowdWisdomTrading historical
trading logs and official strategy parameter definitions were not
available in the supplied workspace.

### Strategy Permutations

The following SMA crossover permutations were evaluated:

- SMA 5/20
- SMA 10/30
- SMA 20/50
- SMA 50/100

The strategy generates a signal using the closing price of day T
and executes the corresponding trade at the next trading day's
open. This avoids using the next day's price to create the signal.

### Macro Data

High-importance United States economic-calendar events were
obtained from the existing Apify dataset. The project uses event
timestamps, actual values, forecast values and calculated
actual-minus-forecast surprise values.

### Macro News

Recent macroeconomic news was collected through Tavily. Articles
were classified into macro themes including:

- Inflation
- Monetary Policy
- Employment
- Growth
- Financial Markets

### Feature Engineering

Macro features include event counts, high-impact event counts,
macro surprises, positive/negative surprise counts and
theme-specific event counts over multiple lookback windows.

### Machine Learning

The model predicts trade outcomes using chronological
walk-forward validation. Random train/test splitting is avoided
because it can introduce temporal leakage in financial data.

### Trading Assumptions

The proxy strategy uses normalized one-unit trade P&L.
Transaction costs, slippage and market impact are not included
in this MVP implementation and should be added before any
production or live-trading use.

## 6. Limitations

1. **Proxy trading data:** The supplied workspace did not contain
   official historical trading logs or official strategy
   definitions, so SPY and SMA crossover strategies were used as
   a transparent proxy.

2. **Small evaluation samples:** Some strategy permutations have
   relatively few test trades. Performance statistics can
   therefore be unstable.

3. **Transaction costs:** The current proxy evaluation does not
   include commissions, slippage or market impact.

4. **Sentiment model:** News sentiment is based on a deterministic
   keyword approach rather than a separately validated financial
   language model.

5. **Macro-news coverage:** Tavily search results represent
   retrieved public news and should not be treated as a complete
   representation of every macroeconomic information source.

6. **No guarantee of future performance:** Historical or
   backtested results do not guarantee future trading results.

7. **Further validation:** A production version should use the
   actual trading logs, official strategy permutations and
   transaction-cost assumptions supplied by CrowdWisdomTrading.

## 7. Generated Project Outputs

Important generated files include:

```text
data/processed/strategy_trades.csv
data/processed/model_dataset.csv
data/processed/walk_forward_predictions.csv
data/processed/walk_forward_report.csv
data/processed/strategy_evaluation.csv

data/processed/macro_news.csv
data/processed/macro_news.json
data/processed/macro_news_sentiment.csv
data/processed/macro_sentiment_matrix.csv
data/processed/macro_sentiment_summary.json
data/processed/macro_outlook.md

data/processed/macro_sentiment_change_matrix.png
data/processed/macro_sentiment_recent.png


## 8. Conclusion

The completed pipeline demonstrates an end-to-end macro-aware trading workflow: market-data ingestion, strategy trade generation, economic-event ingestion, macro feature engineering, recent-news analysis, walk-forward machine learning, strategy filtering and risk/performance evaluation.

The most important next step for a production-grade implementation is to replace the proxy dataset with the official historical trading logs and official strategy parameter permutations, then rerun the complete pipeline with realistic transaction costs and execution assumptions.

---

*Generated automatically by `scripts/build_final_report.py`.*