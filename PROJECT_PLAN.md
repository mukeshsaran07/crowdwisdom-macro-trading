# Project Plan: CrowdWisdomTrading Quantitative Data Scientist MVP

## Project Objective
Build a simple, professional Minimum Viable Product (MVP) to demonstrate quantitative data science skills within a 3-day timeline. The project involves gathering macroeconomic data, engineering features, storing data in SQLite, and predicting future trade P&L or win rate using a chronological walk-forward validation strategy. Finally, evaluate the performance using standard trading metrics (Sharpe, Sortino, Max Drawdown, Win Rate) and generate visualizations and reports.

## Proposed Architecture
- **Data Ingestion**: Use Apify for macroeconomic event data (last 180 days) and Exa/Tavily for recent economic news.
- **Storage**: SQLite database (`macro_trading.db`) using SQLAlchemy to store and join historical trading logs, macro events, and news.
- **Data Processing & Feature Engineering**: Python, pandas, scikit-learn for cleaning and constructing time-based and macroeconomic features.
- **Modeling**: Scikit-learn models (e.g., Random Forest or Gradient Boosting) evaluated using chronological walk-forward (out-of-sample) validation.
- **Metrics & Visualization**: Matplotlib and pandas to compute standard trading metrics and generate visualizations showing macro sentiment changes.
- **Reporting**: Final markdown/PDF evaluation report and a one-page macroeconomic outlook.
- **Environment**: Python, `.env` for secrets, and a fully reproducible GitHub repository.

## Required Data
1. **Historical Trading Logs**: Needs to be provided or acquired (currently MISSING in workspace).
2. **Strategy Names/Parameter Permutations**: Needs to be provided alongside trading logs (currently MISSING).
3. **Macroeconomic Event Data**: To be scraped using Apify for the last ~180 days.
4. **Economic News**: To be retrieved using Tavily/Exa APIs.

## Day 1 Tasks: Setup & Data Ingestion
- Set up project structure, Git repository, and virtual environment.
- Create `.env` and `.env.example`.
- Obtain or clarify the source of the historical trading logs.
- Implement Apify scraper for macroeconomic events.
- Implement Tavily/Exa integration for economic news.
- Design and create the SQLite database schema.

## Day 2 Tasks: Data Processing, Feature Engineering & Modeling
- Store scraped data and trading logs into the SQLite database.
- Join datasets.
- Implement time-based and macroeconomic feature engineering.
- Develop the predictive model (P&L or win rate prediction) using scikit-learn.
- Implement chronological walk-forward validation (no random `train_test_split`).

## Day 3 Tasks: Evaluation, Visualization & Reporting
- Calculate trading metrics: Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Win Rate.
- Generate visual matrix/charts showing macro sentiment changes over time.
- Draft the one-page macroeconomic outlook.
- Draft the final evaluation report.
- Code cleanup, documentation, and final repository review.
