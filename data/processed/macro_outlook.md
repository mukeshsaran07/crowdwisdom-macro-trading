# Macro Outlook

**Generated:** 2026-10-08 13:04 UTC
  
**Latest news timestamp:** 2026-10-08 10:00 UTC

## Executive Summary

Recent news sentiment is **neutral** with an average sentiment score of **0.153** across **20** recent articles.

- **Improving themes:** No theme crossed the improvement threshold.
- **Deteriorating themes:** Employment, Monetary Policy

## Theme Analysis

| Theme | Recent 7d | Previous 7d | Change | Direction |
|---|---:|---:|---:|---|
| Employment | -0.074 | 0.200 | -0.274 | Deteriorating |
| Monetary Policy | 0.352 | 1.000 | -0.648 | Deteriorating |
| Financial Markets | 0.000 | N/A | N/A | Insufficient data |
| Growth | 0.209 | N/A | N/A | Insufficient data |
| Inflation | -1.000 | N/A | N/A | Insufficient data |

## Macro Event Context

- Macro events available in SQLite: **283**
- Average macro surprise: **-13596.0897**
- Positive surprises: **92**
- Negative surprises: **98**

## Trading Relevance

Macro sentiment is treated as an additional information signal rather than a standalone trading decision. The ML pipeline should combine macro features with strategy and market information.

For walk-forward modeling, only information available before the relevant trade entry should be used. This prevents look-ahead bias.

## Data and Methodology

News was collected using Tavily and stored locally in `data/processed/macro_news.csv`. Sentiment is calculated using a deterministic keyword-based scoring method so the calculation is reproducible and auditable.

This sentiment score should not be interpreted as a financial recommendation or as a validated market forecast.

> **Important:** The trading dataset uses SPY as a transparent market proxy because official historical trading logs and strategy definitions were not available in the supplied workspace. Results should therefore be treated as a methodology demonstration, not as official CrowdWisdomTrading performance.
