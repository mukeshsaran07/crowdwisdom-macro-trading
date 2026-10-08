"""
Create visual macro sentiment charts.

Input:
    data/processed/macro_sentiment_matrix.csv

Outputs:
    data/processed/macro_sentiment_change_matrix.png
    data/processed/macro_sentiment_recent.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "macro_sentiment_matrix.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("Loading macro sentiment matrix...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}\n"
            "Run build_macro_sentiment.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Themes loaded: {len(df)}")

    return df


# ============================================================
# CHART 1 — SENTIMENT CHANGE
# ============================================================

def create_change_chart(df):
    print("\nCreating sentiment change chart...")

    chart_df = df.dropna(
        subset=["sentiment_change_7d"]
    ).copy()

    if chart_df.empty:
        print("No sentiment change data available.")
        return

    chart_df = chart_df.sort_values(
        "sentiment_change_7d"
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        chart_df["theme"],
        chart_df["sentiment_change_7d"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.title(
        "Macro Sentiment Change — Recent 7 Days vs Previous 7 Days"
    )

    plt.xlabel(
        "Sentiment Change"
    )

    plt.ylabel(
        "Macro Theme"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "macro_sentiment_change_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path.relative_to(ROOT_DIR)}"
    )


# ============================================================
# CHART 2 — RECENT SENTIMENT
# ============================================================

def create_recent_chart(df):
    print("\nCreating recent sentiment chart...")

    chart_df = df.dropna(
        subset=["Recent 7d"]
    ).copy()

    if chart_df.empty:
        print("No recent sentiment data available.")
        return

    chart_df = chart_df.sort_values(
        "Recent 7d"
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        chart_df["theme"],
        chart_df["Recent 7d"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.title(
        "Recent Macro Sentiment by Theme"
    )

    plt.xlabel(
        "Average Sentiment Score"
    )

    plt.ylabel(
        "Macro Theme"
    )

    plt.xlim(
        -1,
        1,
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "macro_sentiment_recent.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path.relative_to(ROOT_DIR)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 45)
    print("       MACRO SENTIMENT VISUALIZATION")
    print("=" * 45)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    create_change_chart(df)

    create_recent_chart(df)

    print("\n" + "=" * 45)
    print("Macro visualization completed.")
    print("=" * 45)


if __name__ == "__main__":
    main()