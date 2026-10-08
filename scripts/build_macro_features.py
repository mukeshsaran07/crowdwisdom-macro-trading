
import os
import sqlite3
import pandas as pd
import numpy as np


DB_PATH = "data/macro_trading.db"
OUTPUT_PATH = "data/processed/model_dataset.csv"


def load_data():
    """Load trades and macro events from SQLite."""

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    conn = sqlite3.connect(DB_PATH)

    trades = pd.read_sql_query(
        "SELECT * FROM strategy_trades",
        conn
    )

    macro = pd.read_sql_query(
        "SELECT * FROM macro_events",
        conn
    )

    conn.close()

    return trades, macro


def prepare_data(trades, macro):
    """Clean timestamps and numeric columns."""

    # ---------------------------------------------------------
    # Trade timestamps
    # ---------------------------------------------------------
    trades["entry_time"] = pd.to_datetime(
        trades["entry_time"],
        utc=True,
        errors="coerce"
    )

    trades["exit_time"] = pd.to_datetime(
        trades["exit_time"],
        utc=True,
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Macro timestamps
    # ---------------------------------------------------------
    macro["event_time_utc"] = pd.to_datetime(
        macro["event_time_utc"],
        utc=True,
        errors="coerce"
    )

    # Remove invalid timestamps
    trades = trades.dropna(
        subset=["entry_time"]
    ).copy()

    macro = macro.dropna(
        subset=["event_time_utc"]
    ).copy()

    # ---------------------------------------------------------
    # Numeric macro columns
    # ---------------------------------------------------------
    for col in [
        "actual_value",
        "forecast_value",
        "previous_value",
        "macro_surprise"
    ]:
        if col in macro.columns:
            macro[col] = pd.to_numeric(
                macro[col],
                errors="coerce"
            )

    # ---------------------------------------------------------
    # Text columns
    # ---------------------------------------------------------
    macro["event_name"] = (
        macro["event_name"]
        .fillna("")
        .astype(str)
    )

    macro["importance"] = (
        macro["importance"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    # Sort chronologically
    trades = trades.sort_values(
        "entry_time"
    ).reset_index(drop=True)

    macro = macro.sort_values(
        "event_time_utc"
    ).reset_index(drop=True)

    return trades, macro


def calculate_macro_features(
    trade_time,
    macro
):
    """
    Calculate macro features using ONLY events
    strictly before trade_time.

    This prevents look-ahead bias.
    """

    # ---------------------------------------------------------
    # IMPORTANT:
    # Strictly less than trade time.
    #
    # We NEVER use an event occurring at or after
    # the trade entry.
    # ---------------------------------------------------------
    past = macro[
        macro["event_time_utc"] < trade_time
    ].copy()

    features = {}

    # ---------------------------------------------------------
    # Basic time features
    # ---------------------------------------------------------
    features["trade_hour"] = trade_time.hour
    features["trade_day_of_week"] = trade_time.dayofweek
    features["trade_month"] = trade_time.month
    features["trade_quarter"] = (
        (trade_time.month - 1) // 3 + 1
    )

    # ---------------------------------------------------------
    # No previous macro information
    # ---------------------------------------------------------
    if past.empty:

        features["macro_events_1d"] = 0
        features["macro_events_3d"] = 0
        features["macro_events_7d"] = 0

        features["high_impact_events_1d"] = 0
        features["high_impact_events_3d"] = 0
        features["high_impact_events_7d"] = 0

        features["avg_surprise_1d"] = 0.0
        features["avg_surprise_3d"] = 0.0
        features["avg_surprise_7d"] = 0.0

        features["positive_surprises_7d"] = 0
        features["negative_surprises_7d"] = 0

        features["cpi_events_7d"] = 0
        features["fomc_events_7d"] = 0
        features["employment_events_7d"] = 0

        features["latest_macro_surprise"] = 0.0
        features["days_since_macro_event"] = 999.0

        return features

    # ---------------------------------------------------------
    # Event age
    # ---------------------------------------------------------
    past["event_age_days"] = (
        trade_time - past["event_time_utc"]
    ).dt.total_seconds() / 86400.0

    # ---------------------------------------------------------
    # Event window helper
    # ---------------------------------------------------------
    def window(days):
        return past[
            past["event_age_days"] <= days
        ]

    events_1d = window(1)
    events_3d = window(3)
    events_7d = window(7)

    # ---------------------------------------------------------
    # Event counts
    # ---------------------------------------------------------
    features["macro_events_1d"] = len(
        events_1d
    )

    features["macro_events_3d"] = len(
        events_3d
    )

    features["macro_events_7d"] = len(
        events_7d
    )

    # ---------------------------------------------------------
    # High-impact events
    # ---------------------------------------------------------
    high_1d = events_1d[
        events_1d["importance"]
        .str.contains("high", na=False)
    ]

    high_3d = events_3d[
        events_3d["importance"]
        .str.contains("high", na=False)
    ]

    high_7d = events_7d[
        events_7d["importance"]
        .str.contains("high", na=False)
    ]

    features["high_impact_events_1d"] = len(
        high_1d
    )

    features["high_impact_events_3d"] = len(
        high_3d
    )

    features["high_impact_events_7d"] = len(
        high_7d
    )

    # ---------------------------------------------------------
    # Macro surprise
    # ---------------------------------------------------------
    features["avg_surprise_1d"] = (
        events_1d["macro_surprise"]
        .mean()
        if not events_1d.empty
        else 0.0
    )

    features["avg_surprise_3d"] = (
        events_3d["macro_surprise"]
        .mean()
        if not events_3d.empty
        else 0.0
    )

    features["avg_surprise_7d"] = (
        events_7d["macro_surprise"]
        .mean()
        if not events_7d.empty
        else 0.0
    )

    # ---------------------------------------------------------
    # Positive / negative surprises
    # ---------------------------------------------------------
    surprises_7d = events_7d[
        events_7d["macro_surprise"].notna()
    ]

    features["positive_surprises_7d"] = int(
        (
            surprises_7d["macro_surprise"] > 0
        ).sum()
    )

    features["negative_surprises_7d"] = int(
        (
            surprises_7d["macro_surprise"] < 0
        ).sum()
    )

    # ---------------------------------------------------------
    # Event name based features
    # ---------------------------------------------------------
    names = events_7d["event_name"].str.lower()

    features["cpi_events_7d"] = int(
        names.str.contains(
            "cpi|consumer price",
            regex=True,
            na=False
        ).sum()
    )

    features["fomc_events_7d"] = int(
        names.str.contains(
            "fomc|federal reserve|fed rate",
            regex=True,
            na=False
        ).sum()
    )

    features["employment_events_7d"] = int(
        names.str.contains(
            "nonfarm|payroll|employment|unemployment|jobless",
            regex=True,
            na=False
        ).sum()
    )

    # ---------------------------------------------------------
    # Latest macro surprise
    # ---------------------------------------------------------
    latest = past[
        past["macro_surprise"].notna()
    ].sort_values(
        "event_time_utc"
    )

    if latest.empty:
        features["latest_macro_surprise"] = 0.0
    else:
        features["latest_macro_surprise"] = (
            latest.iloc[-1]["macro_surprise"]
        )

    # ---------------------------------------------------------
    # Days since latest macro event
    # ---------------------------------------------------------
    latest_event_time = past[
        "event_time_utc"
    ].max()

    features["days_since_macro_event"] = (
        trade_time - latest_event_time
    ).total_seconds() / 86400.0

    return features


def build_dataset(trades, macro):
    """Build final ML dataset."""

    rows = []

    print(
        f"Building macro features for "
        f"{len(trades)} trades..."
    )

    for index, trade in trades.iterrows():

        if index % 100 == 0:
            print(
                f"Processing trade "
                f"{index + 1}/{len(trades)}"
            )

        entry_time = trade["entry_time"]

        features = calculate_macro_features(
            entry_time,
            macro
        )

        row = features.copy()

        # -----------------------------------------------------
        # Strategy information
        # -----------------------------------------------------
        if "strategy_id" in trade:
            row["strategy_id"] = (
                trade["strategy_id"]
            )

        if "symbol" in trade:
            row["symbol"] = trade["symbol"]

        if "fast_window" in trade:
            row["fast_window"] = (
                trade["fast_window"]
            )

        if "slow_window" in trade:
            row["slow_window"] = (
                trade["slow_window"]
            )

        if "position" in trade:
            row["position"] = (
                trade["position"]
            )

        # -----------------------------------------------------
        # Trade information
        # -----------------------------------------------------
        row["trade_id"] = trade["trade_id"]

        row["entry_time"] = (
            trade["entry_time"]
        )

        # -----------------------------------------------------
        # TARGET VARIABLES
        # -----------------------------------------------------
        #
        # These are the values we eventually want
        # the ML model to predict.
        # -----------------------------------------------------

        if "pnl" in trade:
            row["target_pnl"] = trade["pnl"]

        if "return_pct" in trade:
            row["target_return_pct"] = (
                trade["return_pct"]
            )

        if "win_flag" in trade:
            row["target_win"] = (
                trade["win_flag"]
            )

        rows.append(row)

    dataset = pd.DataFrame(rows)

    return dataset


def main():

    print("========================================")
    print("       MACRO FEATURE ENGINEERING")
    print("========================================")

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------
    print("\nLoading data from SQLite...")

    trades, macro = load_data()

    print(
        f"Trading records loaded: "
        f"{len(trades)}"
    )

    print(
        f"Macro events loaded: "
        f"{len(macro)}"
    )

    # ---------------------------------------------------------
    # 2. Prepare data
    # ---------------------------------------------------------
    print("\nPreparing timestamps and values...")

    trades, macro = prepare_data(
        trades,
        macro
    )

    print(
        f"Valid trades: "
        f"{len(trades)}"
    )

    print(
        f"Valid macro events: "
        f"{len(macro)}"
    )

    # ---------------------------------------------------------
    # 3. Build features
    # ---------------------------------------------------------
    dataset = build_dataset(
        trades,
        macro
    )

    # ---------------------------------------------------------
    # 4. Clean infinite values
    # ---------------------------------------------------------
    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill only feature missing values.
    numeric_cols = dataset.select_dtypes(
        include=[np.number]
    ).columns

    for col in numeric_cols:

        if col.startswith("target_"):
            continue

        dataset[col] = dataset[col].fillna(0)

    # ---------------------------------------------------------
    # 5. Create output directory
    # ---------------------------------------------------------
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    # ---------------------------------------------------------
    # 6. Save dataset
    # ---------------------------------------------------------
    dataset.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------
    print("\n========================================")
    print("       FEATURE ENGINEERING COMPLETE")
    print("========================================")

    print(
        f"Final rows: {len(dataset)}"
    )

    print(
        f"Final columns: {len(dataset.columns)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print("\nFeatures created:")

    feature_columns = [
        col
        for col in dataset.columns
        if not col.startswith("target_")
    ]

    for col in feature_columns:
        print(f"  - {col}")

    print("\nTarget columns:")

    for col in dataset.columns:
        if col.startswith("target_"):
            print(f"  - {col}")

    print("========================================")


if __name__ == "__main__":
    main()

