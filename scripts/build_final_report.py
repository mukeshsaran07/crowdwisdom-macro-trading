"""
Build the final CrowdWisdomTrading evaluation report.

Inputs:
    data/processed/strategy_evaluation.csv
    data/processed/walk_forward_report.csv
    data/processed/macro_sentiment_matrix.csv
    data/processed/macro_sentiment_summary.json
    data/processed/macro_outlook.md

Outputs:
    data/processed/final_evaluation_report.md

This script does not call Apify or Tavily.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT_DIR / "data" / "processed"

STRATEGY_EVAL_FILE = (
    PROCESSED_DIR / "strategy_evaluation.csv"
)

WALK_FORWARD_FILE = (
    PROCESSED_DIR / "walk_forward_report.csv"
)

SENTIMENT_MATRIX_FILE = (
    PROCESSED_DIR / "macro_sentiment_matrix.csv"
)

SENTIMENT_SUMMARY_FILE = (
    PROCESSED_DIR / "macro_sentiment_summary.json"
)

MACRO_OUTLOOK_FILE = (
    PROCESSED_DIR / "macro_outlook.md"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "final_evaluation_report.md"
)


# ============================================================
# HELPERS
# ============================================================

def load_csv(path: Path, name: str) -> pd.DataFrame:
    """Load CSV safely."""

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"{name}: {len(df)} rows loaded"
    )

    return df


def load_json(path: Path, name: str) -> dict:
    """Load JSON safely."""

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    print(f"{name}: loaded")

    return data


def fmt(value, digits=2) -> str:
    """Format numerical values safely."""

    if pd.isna(value):
        return "N/A"

    return f"{float(value):.{digits}f}"


def fmt_pct(value) -> str:
    """Format decimal as percentage."""

    if pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.2f}%"


def find_column(df: pd.DataFrame, candidates):
    """Find first available column."""

    for column in candidates:
        if column in df.columns:
            return column

    return None


# ============================================================
# LOAD ALL RESULTS
# ============================================================

def load_all_data():

    print("=" * 60)
    print("       LOADING FINAL REPORT DATA")
    print("=" * 60)

    strategy_df = load_csv(
        STRATEGY_EVAL_FILE,
        "Strategy evaluation",
    )

    walk_forward_df = load_csv(
        WALK_FORWARD_FILE,
        "Walk-forward report",
    )

    sentiment_df = load_csv(
        SENTIMENT_MATRIX_FILE,
        "Macro sentiment matrix",
    )

    sentiment_summary = load_json(
        SENTIMENT_SUMMARY_FILE,
        "Macro sentiment summary",
    )

    return (
        strategy_df,
        walk_forward_df,
        sentiment_df,
        sentiment_summary,
    )


# ============================================================
# STRATEGY SECTION
# ============================================================

def build_strategy_section(df: pd.DataFrame) -> str:

    lines = []

    lines.append("## 1. Strategy Evaluation")
    lines.append("")

    lines.append(
        "Four SMA crossover strategy permutations were evaluated "
        "using chronological walk-forward validation."
    )

    lines.append("")

    lines.append(
        "| Rank | Strategy | Test Trades | "
        "Baseline P&L | ML P&L | "
        "Baseline Win Rate | ML Win Rate | "
        "ML Sharpe | ML Sortino | ML Max DD | "
        "Profit Factor | Selection Rate |"
    )

    lines.append(
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for _, row in df.iterrows():

        rank = row.get("rank", "")

        strategy = row.get(
            "strategy_id",
            "Unknown",
        )

        test_trades = row.get(
            "test_trades",
            float("nan"),
        )

        baseline_pnl = row.get(
            "baseline_pnl",
            float("nan"),
        )

        ml_pnl = row.get(
            "ml_pnl",
            float("nan"),
        )

        baseline_win = row.get(
            "baseline_win_rate",
            float("nan"),
        )

        ml_win = row.get(
            "ml_win_rate",
            float("nan"),
        )

        ml_sharpe = row.get(
            "ml_sharpe",
            float("nan"),
        )

        ml_sortino = row.get(
            "ml_sortino",
            float("nan"),
        )

        ml_dd = row.get(
            "ml_max_drawdown",
            float("nan"),
        )

        profit_factor = row.get(
            "ml_profit_factor",
            float("nan"),
        )

        selection_rate = row.get(
            "selection_rate",
            float("nan"),
        )

        lines.append(
            f"| {rank} | {strategy} | "
            f"{fmt(test_trades, 0)} | "
            f"{fmt(baseline_pnl)} | "
            f"{fmt(ml_pnl)} | "
            f"{fmt_pct(baseline_win)} | "
            f"{fmt_pct(ml_win)} | "
            f"{fmt(ml_sharpe, 4)} | "
            f"{fmt(ml_sortino, 4)} | "
            f"{fmt(ml_dd)} | "
            f"{fmt(profit_factor, 4)} | "
            f"{fmt_pct(selection_rate)} |"
        )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# RECOMMENDED STRATEGY
# ============================================================

def build_recommendation(df: pd.DataFrame) -> str:

    lines = []

    lines.append("## 2. Model-Selected Strategy")
    lines.append("")

    if df.empty:
        lines.append(
            "No strategy evaluation results were available."
        )
        return "\n".join(lines)

    # Prefer the ranking produced by evaluate_strategies.py.
    if "rank" in df.columns:
        best = df.sort_values(
            "rank"
        ).iloc[0]
    else:
        best = df.sort_values(
            "ml_sharpe",
            ascending=False,
        ).iloc[0]

    strategy = best.get(
        "strategy_id",
        "Unknown",
    )

    ml_pnl = best.get(
        "ml_pnl",
        float("nan"),
    )

    ml_win = best.get(
        "ml_win_rate",
        float("nan"),
    )

    sharpe = best.get(
        "ml_sharpe",
        float("nan"),
    )

    sortino = best.get(
        "ml_sortino",
        float("nan"),
    )

    max_dd = best.get(
        "ml_max_drawdown",
        float("nan"),
    )

    profit_factor = best.get(
        "ml_profit_factor",
        float("nan"),
    )

    selection_rate = best.get(
        "selection_rate",
        float("nan"),
    )

    lines.append(
        f"The evaluation ranking identifies **{strategy}** "
        "as the top-performing evaluated permutation under "
        "the ML-filtered walk-forward results."
    )

    lines.append("")

    lines.append(
        f"- ML-filtered P&L: **{fmt(ml_pnl)}**"
    )

    lines.append(
        f"- ML-filtered win rate: **{fmt_pct(ml_win)}**"
    )

    lines.append(
        f"- Sharpe ratio: **{fmt(sharpe, 4)}**"
    )

    lines.append(
        f"- Sortino ratio: **{fmt(sortino, 4)}**"
    )

    lines.append(
        f"- Maximum drawdown: **{fmt(max_dd)}**"
    )

    lines.append(
        f"- Profit factor: **{fmt(profit_factor, 4)}**"
    )

    lines.append(
        f"- Trade selection rate: **{fmt_pct(selection_rate)}**"
    )

    lines.append("")

    lines.append(
        "**Important:** These results are evaluation results "
        "on the project proxy dataset. They should not be "
        "interpreted as guaranteed future trading performance."
    )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# WALK-FORWARD SECTION
# ============================================================

def build_walk_forward_section(
    df: pd.DataFrame,
) -> str:

    lines = []

    lines.append("## 3. Walk-Forward Validation")
    lines.append("")

    lines.append(
        "The model uses chronological walk-forward evaluation "
        "rather than a random train/test split. This preserves "
        "the temporal ordering of market information."
    )

    lines.append("")

    lines.append(
        "| Strategy | Test Samples | MAE | RMSE | "
        "Accuracy | Precision | Recall | F1 |"
    )

    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for _, row in df.iterrows():

        strategy = row.get(
            "strategy_id",
            "Unknown",
        )

        # Different versions of the training script may use
        # different column names.
        test_count = row.get(
            "test_count",
            row.get(
                "test_trades",
                float("nan"),
            ),
        )

        mae = row.get(
            "mae",
            float("nan"),
        )

        rmse = row.get(
            "rmse",
            float("nan"),
        )

        accuracy = row.get(
            "accuracy",
            float("nan"),
        )

        precision = row.get(
            "precision",
            float("nan"),
        )

        recall = row.get(
            "recall",
            float("nan"),
        )

        f1 = row.get(
            "f1",
            float("nan"),
        )

        lines.append(
            f"| {strategy} | "
            f"{fmt(test_count, 0)} | "
            f"{fmt(mae, 4)} | "
            f"{fmt(rmse, 4)} | "
            f"{fmt_pct(accuracy)} | "
            f"{fmt_pct(precision)} | "
            f"{fmt_pct(recall)} | "
            f"{fmt_pct(f1)} |"
        )

    lines.append("")

    lines.append(
        "The model is therefore evaluated using information "
        "available up to each prediction point, reducing the "
        "risk of look-ahead leakage."
    )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# MACRO SECTION
# ============================================================

def build_macro_section(
    sentiment_df: pd.DataFrame,
    sentiment_summary: dict,
) -> str:

    lines = []

    lines.append("## 4. Macro News and Sentiment Analysis")
    lines.append("")

    total_articles = sentiment_summary.get(
        "total_article_count",
        0,
    )

    recent_articles = sentiment_summary.get(
        "recent_article_count",
        0,
    )

    overall_label = sentiment_summary.get(
        "overall_recent_label",
        "Unknown",
    )

    overall_score = sentiment_summary.get(
        "overall_recent_sentiment",
        0,
    )

    lines.append(
        f"Tavily returned **{total_articles} unique news articles** "
        f"for the macro-news analysis. "
        f"**{recent_articles}** articles fall within the most "
        f"recent seven-day window."
    )

    lines.append("")

    lines.append(
        f"Overall recent news sentiment was "
        f"**{overall_label} ({fmt(overall_score, 3)})**."
    )

    lines.append("")

    lines.append(
        "| Theme | Recent 7d | Previous 7d | "
        "Change | Direction |"
    )

    lines.append(
        "|---|---:|---:|---:|---|"
    )

    for _, row in sentiment_df.iterrows():

        theme = row.get(
            "theme",
            "Unknown",
        )

        recent = row.get(
            "Recent 7d",
            float("nan"),
        )

        previous = row.get(
            "Previous 7d",
            float("nan"),
        )

        change = row.get(
            "sentiment_change_7d",
            float("nan"),
        )

        direction = row.get(
            "change_label",
            "Unknown",
        )

        lines.append(
            f"| {theme} | "
            f"{fmt(recent, 3)} | "
            f"{fmt(previous, 3)} | "
            f"{fmt(change, 3)} | "
            f"{direction} |"
        )

    lines.append("")

    lines.append(
        "The sentiment score is a transparent, deterministic "
        "keyword-based indicator. It is used as an exploratory "
        "macro feature and should not be interpreted as a "
        "standalone financial forecast."
    )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# DATA METHODOLOGY
# ============================================================

def build_methodology_section() -> str:

    return """## 5. Data and Methodology

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
"""


# ============================================================
# LIMITATIONS
# ============================================================

def build_limitations_section() -> str:

    return """## 6. Limitations

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
"""


# ============================================================
# PROJECT OUTPUTS
# ============================================================

def build_outputs_section() -> str:

    return """## 7. Generated Project Outputs

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
"""

def build_report(
    strategy_df: pd.DataFrame,
    walk_forward_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    sentiment_summary: dict,
) -> str:

    lines = []

    lines.append(
        "# CrowdWisdomTrading Macro Trading ML Evaluation"
    )

    lines.append("")

    lines.append(
        "**Project type:** Macro-aware ML trading strategy evaluation"
    )

    lines.append("")

    lines.append(
        "**Status:** Prototype / assessment implementation"
    )

    lines.append("")

    lines.append(
        "> This report is a methodology demonstration using "
        "transparent proxy data where official CrowdWisdomTrading "
        "trading logs and strategy definitions were unavailable "
        "in the supplied workspace."
    )

    lines.append("")

    lines.append("---")
    lines.append("")

    lines.append(
        build_strategy_section(strategy_df)
    )

    lines.append(
        build_recommendation(strategy_df)
    )

    lines.append(
        build_walk_forward_section(
            walk_forward_df
        )
    )

    lines.append(
        build_macro_section(
            sentiment_df,
            sentiment_summary,
        )
    )

    lines.append(
        build_methodology_section()
    )

    lines.append(
        build_limitations_section()
    )

    lines.append(
        build_outputs_section()
    )

    lines.append("")

    lines.append("## 8. Conclusion")
    lines.append("")

    lines.append(
        "The completed pipeline demonstrates an end-to-end "
        "macro-aware trading workflow: market-data ingestion, "
        "strategy trade generation, economic-event ingestion, "
        "macro feature engineering, recent-news analysis, "
        "walk-forward machine learning, strategy filtering and "
        "risk/performance evaluation."
    )

    lines.append("")

    lines.append(
        "The most important next step for a production-grade "
        "implementation is to replace the proxy dataset with the "
        "official historical trading logs and official strategy "
        "parameter permutations, then rerun the complete pipeline "
        "with realistic transaction costs and execution assumptions."
    )

    lines.append("")

    lines.append("---")
    lines.append("")

    lines.append(
        "*Generated automatically by "
        "`scripts/build_final_report.py`.*"
    )

    return "\n".join(lines)


def main():

    print("=" * 60)
    print("       BUILDING FINAL EVALUATION REPORT")
    print("=" * 60)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        strategy_df,
        walk_forward_df,
        sentiment_df,
        sentiment_summary,
    ) = load_all_data()

    report = build_report(
        strategy_df,
        walk_forward_df,
        sentiment_df,
        sentiment_summary,
    )

    OUTPUT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print("")
    print("Final report saved:")
    print(
        f"  {OUTPUT_FILE.relative_to(ROOT_DIR)}"
    )

    print("")
    print("=" * 60)
    print("       FINAL REPORT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()