"""
Build macro news sentiment and macro sentiment change analysis.

Inputs:
    data/processed/macro_news.csv
    data/macro_trading.db

Outputs:
    data/processed/macro_news_sentiment.csv
    data/processed/macro_sentiment_matrix.csv
    data/processed/macro_outlook.md
    data/processed/macro_sentiment_summary.json

No external API calls are made by this script.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DB_PATH = ROOT_DIR / "data" / "macro_trading.db"
NEWS_PATH = ROOT_DIR / "data" / "processed" / "macro_news.csv"

OUTPUT_DIR = ROOT_DIR / "data" / "processed"

NEWS_SENTIMENT_PATH = OUTPUT_DIR / "macro_news_sentiment.csv"
SENTIMENT_MATRIX_PATH = OUTPUT_DIR / "macro_sentiment_matrix.csv"
OUTLOOK_PATH = OUTPUT_DIR / "macro_outlook.md"
SUMMARY_PATH = OUTPUT_DIR / "macro_sentiment_summary.json"


# ============================================================
# THEME KEYWORDS
# ============================================================

THEME_KEYWORDS = {
    "Inflation": [
        "inflation",
        "cpi",
        "consumer price",
        "prices",
        "price pressure",
        "core inflation",
        "pce",
        "ppi",
        "producer price",
    ],
    "Monetary Policy": [
        "federal reserve",
        "fed",
        "fomc",
        "interest rate",
        "interest rates",
        "rate cut",
        "rate hike",
        "rate increase",
        "rate decrease",
        "monetary policy",
        "fed funds",
        "policy rate",
    ],
    "Employment": [
        "employment",
        "jobs",
        "job",
        "payroll",
        "nonfarm",
        "unemployment",
        "labor market",
        "labour market",
        "wages",
        "wage growth",
        "jobless claims",
    ],
    "Growth": [
        "gdp",
        "economic growth",
        "growth",
        "recession",
        "economic activity",
        "business activity",
        "manufacturing",
        "services activity",
        "consumer spending",
        "retail sales",
    ],
    "Financial Markets": [
        "stock market",
        "stocks",
        "equity market",
        "equities",
        "s&p 500",
        "sp500",
        "nasdaq",
        "market volatility",
        "financial markets",
        "market outlook",
    ],
}


# ============================================================
# SENTIMENT WORDS
# ============================================================

POSITIVE_WORDS = {
    "strong",
    "stronger",
    "improve",
    "improved",
    "improves",
    "improving",
    "growth",
    "growing",
    "recovery",
    "recover",
    "recovered",
    "robust",
    "resilient",
    "resilience",
    "optimistic",
    "optimism",
    "positive",
    "upside",
    "better",
    "increase",
    "increased",
    "increases",
    "increasing",
    "gain",
    "gains",
    "gained",
    "surge",
    "surged",
    "surging",
    "beat",
    "beats",
    "beating",
    "exceed",
    "exceeded",
    "exceeds",
    "healthy",
    "solid",
    "stable",
    "stabilize",
    "stabilized",
}

NEGATIVE_WORDS = {
    "weak",
    "weaker",
    "worsen",
    "worsened",
    "worsening",
    "decline",
    "declined",
    "declines",
    "declining",
    "slowdown",
    "slow",
    "slower",
    "recession",
    "recessionary",
    "risk",
    "risks",
    "risky",
    "negative",
    "downside",
    "worse",
    "decrease",
    "decreased",
    "decreases",
    "decreasing",
    "fall",
    "falls",
    "fell",
    "drop",
    "dropped",
    "dropping",
    "miss",
    "missed",
    "misses",
    "high",
    "higher",
    "elevated",
    "pressure",
    "pressures",
    "uncertainty",
    "uncertain",
    "volatile",
    "volatility",
}


# Words that are especially important for macro interpretation.
# They receive a little more weight than generic words.
STRONG_POSITIVE = {
    "recovery",
    "recover",
    "robust",
    "resilient",
    "surge",
    "surged",
    "strong",
    "stronger",
    "beat",
    "exceeded",
    "exceeds",
}

STRONG_NEGATIVE = {
    "recession",
    "recessionary",
    "slowdown",
    "weak",
    "weaker",
    "worsening",
    "uncertainty",
    "volatile",
    "volatility",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_text(value) -> str:
    """Convert a value to normalized lowercase text."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).lower()

    # Remove URLs.
    text = re.sub(r"https?://\S+", " ", text)

    # Keep letters/numbers and spaces.
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())


def detect_theme(title: str, content: str = "") -> str:
    """
    Detect the main macro theme from article title/content.

    Priority is based on the number of matching keywords.
    """

    combined = clean_text(f"{title} {content}")

    if not combined:
        return "General Macro"

    scores = {}

    for theme, keywords in THEME_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            keyword_clean = clean_text(keyword)

            if keyword_clean in combined:
                # Exact phrase matches get a little more weight.
                if " " in keyword_clean:
                    score += 2
                else:
                    score += 1

        scores[theme] = score

    best_theme = max(scores, key=scores.get)

    if scores[best_theme] == 0:
        return "General Macro"

    return best_theme


def sentiment_score(text: str) -> float:
    """
    Lightweight deterministic macro sentiment score.

    Range is approximately [-1, +1].

    This is intentionally transparent rather than using an external
    black-box sentiment model.
    """

    text_clean = clean_text(text)

    if not text_clean:
        return 0.0

    words = tokenize(text_clean)

    positive = 0.0
    negative = 0.0

    for word in words:
        if word in POSITIVE_WORDS:
            positive += 1.0

        if word in NEGATIVE_WORDS:
            negative += 1.0

        if word in STRONG_POSITIVE:
            positive += 0.75

        if word in STRONG_NEGATIVE:
            negative += 0.75

    total = positive + negative

    if total == 0:
        return 0.0

    score = (positive - negative) / total

    return float(np.clip(score, -1.0, 1.0))


def sentiment_label(score: float) -> str:
    """Convert numerical sentiment into a simple label."""

    if score >= 0.20:
        return "Positive"

    if score <= -0.20:
        return "Negative"

    return "Neutral"


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse timestamps safely as UTC."""

    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


# ============================================================
# LOAD NEWS
# ============================================================

def load_news() -> pd.DataFrame:
    """Load previously collected Tavily news."""

    print("Loading Tavily news...")

    if not NEWS_PATH.exists():
        raise FileNotFoundError(
            f"News file not found: {NEWS_PATH}\n"
            "Run scripts/fetch_macro_news.py first."
        )

    df = pd.read_csv(NEWS_PATH)

    print(f"News articles loaded: {len(df)}")

    if df.empty:
        raise ValueError("macro_news.csv is empty.")

    return df


# ============================================================
# LOAD MACRO EVENTS
# ============================================================

def load_macro_events() -> pd.DataFrame:
    """Load macro events from SQLite for additional context."""

    print("Loading macro events from SQLite...")

    if not DB_PATH.exists():
        print("SQLite database not found. Continuing without macro events.")
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)

    try:
        query = """
        SELECT
            event_id,
            event_time_utc,
            event_name,
            importance,
            actual_value,
            forecast_value,
            macro_surprise,
            currency,
            country
        FROM macro_events
        ORDER BY event_time_utc
        """

        df = pd.read_sql_query(query, conn)

    finally:
        conn.close()

    print(f"Macro events loaded: {len(df)}")

    return df


# ============================================================
# PREPARE NEWS
# ============================================================

def prepare_news(df: pd.DataFrame) -> pd.DataFrame:
    """Clean news data and calculate sentiment/theme."""

    print("\nPreparing news data...")

    # Support possible column naming differences.
    title_col = None

    for candidate in ["title", "name", "headline"]:
        if candidate in df.columns:
            title_col = candidate
            break

    if title_col is None:
        raise ValueError(
            "Could not find a title column in macro_news.csv."
        )

    df["title"] = df[title_col].fillna("").astype(str)

    if "content" in df.columns:
        df["content"] = df["content"].fillna("").astype(str)
    elif "description" in df.columns:
        df["content"] = df["description"].fillna("").astype(str)
    else:
        df["content"] = ""

    if "published" in df.columns:
        df["published_at"] = parse_datetime_series(df["published"])
    elif "published_date" in df.columns:
        df["published_at"] = parse_datetime_series(df["published_date"])
    elif "date" in df.columns:
        df["published_at"] = parse_datetime_series(df["date"])
    else:
        df["published_at"] = pd.NaT

    if "url" not in df.columns:
        df["url"] = ""

    if "source" not in df.columns:
        df["source"] = ""

    # Combine title + content for analysis.
    df["analysis_text"] = (
        df["title"].fillna("")
        + " "
        + df["content"].fillna("")
    )

    # Theme.
    df["theme"] = df.apply(
        lambda row: detect_theme(
            row["title"],
            row["content"],
        ),
        axis=1,
    )

    # Sentiment.
    df["sentiment_score"] = df["analysis_text"].apply(
        sentiment_score
    )

    df["sentiment_label"] = df["sentiment_score"].apply(
        sentiment_label
    )

    # Date.
    df["published_date"] = df["published_at"].dt.date

    # Sort newest first.
    df = df.sort_values(
        "published_at",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return df


# ============================================================
# BUILD SENTIMENT MATRIX
# ============================================================

def build_sentiment_matrix(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a theme x time-period sentiment matrix.

    Periods:
        Recent 7d
        Previous 7d
        Older
    """

    print("\nBuilding macro sentiment matrix...")

    valid_dates = news_df["published_at"].dropna()

    if valid_dates.empty:
        print("No valid publication dates. Using article-order analysis.")

        news_df = news_df.copy()
        news_df["period"] = "Recent"

    else:
        latest_date = valid_dates.max()

        recent_start = latest_date - pd.Timedelta(days=7)
        previous_start = latest_date - pd.Timedelta(days=14)

        def assign_period(dt):
            if pd.isna(dt):
                return "Older"

            if dt >= recent_start:
                return "Recent 7d"

            if dt >= previous_start:
                return "Previous 7d"

            return "Older"

        news_df = news_df.copy()

        news_df["period"] = news_df["published_at"].apply(
            assign_period
        )

    grouped = (
        news_df
        .groupby(["theme", "period"], dropna=False)
        .agg(
            article_count=("title", "count"),
            average_sentiment=("sentiment_score", "mean"),
            positive_articles=(
                "sentiment_label",
                lambda x: int((x == "Positive").sum()),
            ),
            neutral_articles=(
                "sentiment_label",
                lambda x: int((x == "Neutral").sum()),
            ),
            negative_articles=(
                "sentiment_label",
                lambda x: int((x == "Negative").sum()),
            ),
        )
        .reset_index()
    )

    # Pivot recent/previous sentiment.
    pivot = grouped.pivot(
        index="theme",
        columns="period",
        values="average_sentiment",
    ).reset_index()

    # Make sure expected columns exist.
    for col in ["Recent 7d", "Previous 7d", "Older"]:
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot["sentiment_change_7d"] = (
        pivot["Recent 7d"] - pivot["Previous 7d"]
    )

    def change_label(value):
        if pd.isna(value):
            return "Insufficient data"

        if value >= 0.20:
            return "Improving"

        if value <= -0.20:
            return "Deteriorating"

        return "Stable"

    pivot["change_label"] = pivot["sentiment_change_7d"].apply(
        change_label
    )

    # Overall recent sentiment.
    recent = (
        news_df[news_df["period"] == "Recent 7d"]
        if "period" in news_df.columns
        else news_df
    )

    if not recent.empty:
        overall_recent = recent["sentiment_score"].mean()
    else:
        overall_recent = np.nan

    pivot["overall_recent_sentiment"] = overall_recent

    return pivot.sort_values(
        "sentiment_change_7d",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)


# ============================================================
# MACRO EVENT SUMMARY
# ============================================================

def build_macro_event_summary(events_df: pd.DataFrame) -> dict:
    """Summarize recent macro event surprises."""

    if events_df.empty:
        return {
            "event_count": 0,
            "average_surprise": None,
            "positive_surprises": 0,
            "negative_surprises": 0,
        }

    df = events_df.copy()

    if "event_time_utc" in df.columns:
        df["event_time_utc"] = parse_datetime_series(
            df["event_time_utc"]
        )

    if "macro_surprise" in df.columns:
        surprises = pd.to_numeric(
            df["macro_surprise"],
            errors="coerce",
        ).dropna()
    else:
        surprises = pd.Series(dtype=float)

    return {
        "event_count": int(len(df)),
        "average_surprise": (
            float(surprises.mean())
            if not surprises.empty
            else None
        ),
        "positive_surprises": int(
            (surprises > 0).sum()
        ),
        "negative_surprises": int(
            (surprises < 0).sum()
        ),
    }


# ============================================================
# BUILD MACRO OUTLOOK
# ============================================================

def build_outlook(
    news_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    event_summary: dict,
) -> str:
    """Create a concise one-page macro outlook in Markdown."""

    generated_at = pd.Timestamp.now(tz="UTC")

    valid_dates = news_df["published_at"].dropna()

    if not valid_dates.empty:
        latest_news_date = valid_dates.max()
    else:
        latest_news_date = None

    recent_news = news_df[
        news_df["period"] == "Recent 7d"
    ].copy()

    # Overall sentiment.
    if not recent_news.empty:
        overall_score = float(
            recent_news["sentiment_score"].mean()
        )
    else:
        overall_score = 0.0

    overall_label = sentiment_label(overall_score)

    # Find strongest improving/deteriorating themes.
    improving = matrix_df[
        matrix_df["change_label"] == "Improving"
    ]

    deteriorating = matrix_df[
        matrix_df["change_label"] == "Deteriorating"
    ]

    stable = matrix_df[
        matrix_df["change_label"] == "Stable"
    ]

    # Recent article count.
    recent_count = len(recent_news)

    lines = []

    lines.append("# Macro Outlook")
    lines.append("")
    lines.append(
        f"**Generated:** {generated_at.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    if latest_news_date is not None:
        lines.append(
            f"  \n**Latest news timestamp:** "
            f"{latest_news_date.strftime('%Y-%m-%d %H:%M UTC')}"
        )

    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")

    lines.append(
        f"Recent news sentiment is **{overall_label.lower()}** "
        f"with an average sentiment score of "
        f"**{overall_score:.3f}** across "
        f"**{recent_count}** recent articles."
    )

    lines.append("")

    # Improving themes.
    if not improving.empty:
        themes = ", ".join(
            improving["theme"].astype(str).tolist()
        )

        lines.append(
            f"- **Improving themes:** {themes}"
        )
    else:
        lines.append(
            "- **Improving themes:** No theme crossed the "
            "improvement threshold."
        )

    # Deteriorating themes.
    if not deteriorating.empty:
        themes = ", ".join(
            deteriorating["theme"].astype(str).tolist()
        )

        lines.append(
            f"- **Deteriorating themes:** {themes}"
        )
    else:
        lines.append(
            "- **Deteriorating themes:** No theme crossed the "
            "deterioration threshold."
        )

    # Stable themes.
    if not stable.empty:
        themes = ", ".join(
            stable["theme"].astype(str).tolist()
        )

        lines.append(
            f"- **Stable themes:** {themes}"
        )

    lines.append("")

    lines.append("## Theme Analysis")
    lines.append("")

    lines.append(
        "| Theme | Recent 7d | Previous 7d | Change | Direction |"
    )
    lines.append(
        "|---|---:|---:|---:|---|"
    )

    for _, row in matrix_df.iterrows():

        recent = row.get("Recent 7d")
        previous = row.get("Previous 7d")
        change = row.get("sentiment_change_7d")
        direction = row.get("change_label", "Unknown")

        recent_text = (
            f"{recent:.3f}"
            if pd.notna(recent)
            else "N/A"
        )

        previous_text = (
            f"{previous:.3f}"
            if pd.notna(previous)
            else "N/A"
        )

        change_text = (
            f"{change:+.3f}"
            if pd.notna(change)
            else "N/A"
        )

        lines.append(
            f"| {row['theme']} | "
            f"{recent_text} | "
            f"{previous_text} | "
            f"{change_text} | "
            f"{direction} |"
        )

    lines.append("")

    lines.append("## Macro Event Context")
    lines.append("")

    lines.append(
        f"- Macro events available in SQLite: "
        f"**{event_summary['event_count']}**"
    )

    if event_summary["average_surprise"] is not None:
        lines.append(
            f"- Average macro surprise: "
            f"**{event_summary['average_surprise']:.4f}**"
        )

    lines.append(
        f"- Positive surprises: "
        f"**{event_summary['positive_surprises']}**"
    )

    lines.append(
        f"- Negative surprises: "
        f"**{event_summary['negative_surprises']}**"
    )

    lines.append("")

    lines.append("## Trading Relevance")
    lines.append("")

    lines.append(
        "Macro sentiment is treated as an additional information "
        "signal rather than a standalone trading decision. "
        "The ML pipeline should combine macro features with "
        "strategy and market information."
    )

    lines.append("")

    lines.append(
        "For walk-forward modeling, only information available "
        "before the relevant trade entry should be used. "
        "This prevents look-ahead bias."
    )

    lines.append("")

    lines.append("## Data and Methodology")
    lines.append("")

    lines.append(
        "News was collected using Tavily and stored locally in "
        "`data/processed/macro_news.csv`. Sentiment is calculated "
        "using a deterministic keyword-based scoring method so "
        "the calculation is reproducible and auditable."
    )

    lines.append("")

    lines.append(
        "This sentiment score should not be interpreted as a "
        "financial recommendation or as a validated market "
        "forecast."
    )

    lines.append("")

    lines.append(
        "> **Important:** The trading dataset uses SPY as a "
        "transparent market proxy because official historical "
        "trading logs and strategy definitions were not available "
        "in the supplied workspace. Results should therefore be "
        "treated as a methodology demonstration, not as official "
        "CrowdWisdomTrading performance."
    )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# SAVE SUMMARY
# ============================================================

def build_summary(
    news_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    event_summary: dict,
) -> dict:
    """Build machine-readable summary JSON."""

    recent_news = news_df[
        news_df["period"] == "Recent 7d"
    ]

    if recent_news.empty:
        overall_score = 0.0
    else:
        overall_score = float(
            recent_news["sentiment_score"].mean()
        )

    theme_summary = []

    for _, row in matrix_df.iterrows():

        theme_summary.append(
            {
                "theme": str(row["theme"]),
                "recent_7d_sentiment": (
                    None
                    if pd.isna(row.get("Recent 7d"))
                    else float(row["Recent 7d"])
                ),
                "previous_7d_sentiment": (
                    None
                    if pd.isna(row.get("Previous 7d"))
                    else float(row["Previous 7d"])
                ),
                "sentiment_change": (
                    None
                    if pd.isna(row.get("sentiment_change_7d"))
                    else float(row["sentiment_change_7d"])
                ),
                "direction": str(
                    row.get("change_label", "Unknown")
                ),
            }
        )

    return {
        "overall_recent_sentiment": overall_score,
        "overall_recent_label": sentiment_label(
            overall_score
        ),
        "recent_article_count": int(len(recent_news)),
        "total_article_count": int(len(news_df)),
        "theme_count": int(news_df["theme"].nunique()),
        "macro_event_summary": event_summary,
        "themes": theme_summary,
        "method": (
            "Deterministic keyword-based sentiment scoring "
            "with theme classification."
        ),
        "lookback_period": "7 days",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 40)
    print("      MACRO SENTIMENT ANALYSIS")
    print("=" * 40)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load news
    # --------------------------------------------------------

    news_df = load_news()

    # --------------------------------------------------------
    # 2. Load macro events
    # --------------------------------------------------------

    events_df = load_macro_events()

    # --------------------------------------------------------
    # 3. Prepare news
    # --------------------------------------------------------

    news_df = prepare_news(news_df)

    print(
        f"News sentiment calculated for "
        f"{len(news_df)} articles."
    )

    # --------------------------------------------------------
    # 4. Build sentiment periods
    # --------------------------------------------------------

    valid_dates = news_df["published_at"].dropna()

    if not valid_dates.empty:

        latest_date = valid_dates.max()

        recent_start = latest_date - pd.Timedelta(days=7)
        previous_start = latest_date - pd.Timedelta(days=14)

        def assign_period(dt):

            if pd.isna(dt):
                return "Older"

            if dt >= recent_start:
                return "Recent 7d"

            if dt >= previous_start:
                return "Previous 7d"

            return "Older"

        news_df["period"] = news_df[
            "published_at"
        ].apply(assign_period)

    else:
        news_df["period"] = "Recent 7d"

    # --------------------------------------------------------
    # 5. Save detailed news sentiment
    # --------------------------------------------------------

    output_columns = [
        "title",
        "url",
        "source",
        "published_at",
        "theme",
        "sentiment_score",
        "sentiment_label",
        "period",
    ]

    available_columns = [
        col
        for col in output_columns
        if col in news_df.columns
    ]

    news_output = news_df[
        available_columns
    ].copy()

    news_output.to_csv(
        NEWS_SENTIMENT_PATH,
        index=False,
    )

    print(
        f"News sentiment CSV saved: "
        f"{NEWS_SENTIMENT_PATH.relative_to(ROOT_DIR)}"
    )

    # --------------------------------------------------------
    # 6. Build matrix
    # --------------------------------------------------------

    matrix_df = build_sentiment_matrix(
        news_df
    )

    matrix_df.to_csv(
        SENTIMENT_MATRIX_PATH,
        index=False,
    )

    print(
        f"Sentiment matrix saved: "
        f"{SENTIMENT_MATRIX_PATH.relative_to(ROOT_DIR)}"
    )

    # --------------------------------------------------------
    # 7. Macro event summary
    # --------------------------------------------------------

    event_summary = build_macro_event_summary(
        events_df
    )

    # --------------------------------------------------------
    # 8. Build outlook
    # --------------------------------------------------------

    outlook = build_outlook(
        news_df,
        matrix_df,
        event_summary,
    )

    OUTLOOK_PATH.write_text(
        outlook,
        encoding="utf-8",
    )

    print(
        f"Macro outlook saved: "
        f"{OUTLOOK_PATH.relative_to(ROOT_DIR)}"
    )

    # --------------------------------------------------------
    # 9. Build JSON summary
    # --------------------------------------------------------

    summary = build_summary(
        news_df,
        matrix_df,
        event_summary,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(
        f"Sentiment summary saved: "
        f"{SUMMARY_PATH.relative_to(ROOT_DIR)}"
    )

    # --------------------------------------------------------
    # 10. Print results
    # --------------------------------------------------------

    print("\n" + "=" * 40)
    print("      SENTIMENT SUMMARY")
    print("=" * 40)

    print(
        f"Total articles: "
        f"{len(news_df)}"
    )

    print(
        f"Recent 7-day articles: "
        f"{len(news_df[news_df['period'] == 'Recent 7d'])}"
    )

    overall_score = summary[
        "overall_recent_sentiment"
    ]

    print(
        f"Overall recent sentiment: "
        f"{summary['overall_recent_label']} "
        f"({overall_score:.3f})"
    )

    print("\nTheme sentiment:")

    for _, row in matrix_df.iterrows():

        recent = row.get("Recent 7d")
        change = row.get(
            "sentiment_change_7d"
        )

        recent_text = (
            f"{recent:.3f}"
            if pd.notna(recent)
            else "N/A"
        )

        change_text = (
            f"{change:+.3f}"
            if pd.notna(change)
            else "N/A"
        )

        print(
            f"  - {row['theme']}: "
            f"recent={recent_text}, "
            f"change={change_text}, "
            f"{row['change_label']}"
        )

    print("\n" + "=" * 40)
    print("Macro sentiment analysis completed.")
    print("=" * 40)


if __name__ == "__main__":
    main()