
import os
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from tavily import TavilyClient


# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "data/macro_trading.db"

OUTPUT_CSV = "data/processed/macro_news.csv"

OUTPUT_JSON = "data/processed/macro_news.json"

TAVILY_KEY_NAME = "TAVILY_API_KEY"


# ============================================================
# DATABASE
# ============================================================

def load_recent_macro_events():

    print("Loading recent macro events from SQLite...")

    if not os.path.exists(
        "data/macro_trading.db"
    ):
        raise FileNotFoundError(
            "Database not found: data/macro_trading.db"
        )

    connection = sqlite3.connect(
        DB_PATH
    )

    query = """
        SELECT
            event_time_utc,
            country,
            currency,
            event_name,
            category,
            importance,
            actual_value,
            forecast_value,
            previous_value,
            macro_surprise
        FROM macro_events
        ORDER BY event_time_utc DESC
        LIMIT 30
    """

    df = pd.read_sql_query(
        query,
        connection
    )

    connection.close()

    print(
        f"Recent macro events loaded: {len(df)}"
    )

    return df


# ============================================================
# DETERMINE MACRO THEMES
# ============================================================

def identify_macro_themes(df):

    text = " ".join(
        df["event_name"]
        .dropna()
        .astype(str)
        .tolist()
    ).lower()

    themes = []

    if "cpi" in text or "inflation" in text:
        themes.append("US inflation CPI")

    if (
        "fomc" in text
        or "fed" in text
        or "interest rate" in text
        or "federal funds" in text
    ):
        themes.append("Federal Reserve monetary policy")

    if (
        "nonfarm" in text
        or "employment" in text
        or "unemployment" in text
        or "payroll" in text
    ):
        themes.append("US employment")

    if (
        "gdp" in text
        or "gross domestic" in text
    ):
        themes.append("US economic growth")

    if (
        "retail sales" in text
        or "consumer confidence" in text
        or "consumer sentiment" in text
    ):
        themes.append("US consumer activity")

    if (
        "ppi" in text
        or "producer price" in text
    ):
        themes.append("US producer prices")

    if (
        "pce" in text
        or "personal consumption" in text
    ):
        themes.append("US PCE inflation")

    if not themes:
        themes = [
            "US macroeconomic conditions",
            "US monetary policy",
            "US economic outlook"
        ]

    return themes


# ============================================================
# BUILD SEARCH QUERIES
# ============================================================

def build_queries(themes):

    queries = []

    queries.append(
        "latest US macroeconomic news "
        "Federal Reserve inflation employment "
        "October 2026"
    )

    queries.append(
        "latest US CPI inflation Federal Reserve "
        "interest rates economic outlook October 2026"
    )

    queries.append(
        "latest US employment jobs payrolls "
        "economic outlook October 2026"
    )

    queries.append(
        "latest US stock market macroeconomic outlook "
        "Federal Reserve October 2026"
    )

    # Add theme-specific queries
    for theme in themes[:3]:

        queries.append(
            f"latest {theme} US economic news "
            "October 2026"
        )

    # Remove duplicates while preserving order
    unique_queries = []

    for query in queries:

        if query not in unique_queries:
            unique_queries.append(query)

    return unique_queries


# ============================================================
# TAVILY SEARCH
# ============================================================

def search_tavily():

    api_key = os.getenv(
        TAVILY_KEY_NAME
    )

    if not api_key:

        raise RuntimeError(
            "TAVILY_API_KEY is missing "
            "from .env"
        )

    client = TavilyClient(
        api_key=api_key
    )

    # --------------------------------------------------------
    # Load macro events
    # --------------------------------------------------------

    macro_events = load_recent_macro_events()

    if macro_events.empty:

        print(
            "Warning: No macro events found."
        )

        themes = [
            "US inflation",
            "Federal Reserve",
            "US employment"
        ]

    else:

        themes = identify_macro_themes(
            macro_events
        )

    print("\nMacro themes:")

    for theme in themes:
        print(
            f"  - {theme}"
        )

    # --------------------------------------------------------
    # Build queries
    # --------------------------------------------------------

    queries = build_queries(
        themes
    )

    print(
        f"\nTavily queries: {len(queries)}"
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    all_results = []

    seen_urls = set()

    for index, query in enumerate(
        queries,
        start=1
    ):

        print(
            f"\nSearching {index}/{len(queries)}:"
        )

        print(query)

        try:

            response = client.search(
                query=query,
                search_depth="advanced",
                topic="news",
                max_results=5,
                include_answer=False,
                include_raw_content=False
            )

            results = response.get(
                "results",
                []
            )

            print(
                f"Results returned: {len(results)}"
            )

            for result in results:

                url = result.get(
                    "url",
                    ""
                )

                if not url:
                    continue

                if url in seen_urls:
                    continue

                seen_urls.add(url)

                all_results.append(
                    {
                        "query": query,
                        "title": result.get(
                            "title",
                            ""
                        ),
                        "url": url,
                        "content": result.get(
                            "content",
                            ""
                        ),
                        "score": result.get(
                            "score",
                            None
                        ),
                        "published_date": result.get(
                            "published_date",
                            None
                        ),
                        "source": result.get(
                            "url",
                            ""
                        ),
                    }
                )

        except Exception as error:

            print(
                f"Warning: Tavily query failed: "
                f"{error}"
            )

    return macro_events, themes, all_results


# ============================================================
# SAVE NEWS
# ============================================================

def save_results(
    macro_events,
    themes,
    news_results
):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    if news_results:

        news_df = pd.DataFrame(
            news_results
        )

        news_df.insert(
            0,
            "retrieved_at_utc",
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        news_df.to_csv(
            OUTPUT_CSV,
            index=False
        )

        print(
            f"\nNews CSV saved: "
            f"{OUTPUT_CSV}"
        )

    else:

        news_df = pd.DataFrame()

        print(
            "\nNo news results to save."
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    output = {

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "macro_themes":
            themes,

        "macro_event_count":
            len(macro_events),

        "news_result_count":
            len(news_results),

        "news":
            news_results
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    print(
        f"News JSON saved: "
        f"{OUTPUT_JSON}"
    )

    return news_df


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    macro_events,
    themes,
    news_df
):

    print(
        "\n========================================"
    )

    print(
        "       MACRO NEWS INGESTION SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Recent macro events: "
        f"{len(macro_events)}"
    )

    print(
        f"Macro themes detected: "
        f"{len(themes)}"
    )

    print(
        f"Unique news articles: "
        f"{len(news_df)}"
    )

    print("\nThemes:")

    for theme in themes:

        print(
            f"  - {theme}"
        )

    print("\nTop news:")

    if news_df.empty:

        print(
            "  No news articles found."
        )

    else:

        for _, row in news_df.head(
            10
        ).iterrows():

            title = str(
                row.get(
                    "title",
                    ""
                )
            )

            published = row.get(
                "published_date",
                None
            )

            print(
                f"  - {title}"
            )

            if published:

                print(
                    f"    Published: "
                    f"{published}"
                )

    print(
        "\n========================================"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "========================================"
    )

    print(
        "       MACRO NEWS INGESTION"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Load environment
    # --------------------------------------------------------

    load_dotenv()

    # --------------------------------------------------------
    # Search Tavily
    # --------------------------------------------------------

    (
        macro_events,
        themes,
        news_results
    ) = search_tavily()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    news_df = save_results(
        macro_events,
        themes,
        news_results
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        macro_events,
        themes,
        news_df
    )

    print(
        "\nMacro news ingestion completed."
    )


if __name__ == "__main__":
    main()

