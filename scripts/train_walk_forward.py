
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


INPUT_PATH = "data/processed/model_dataset.csv"

PREDICTIONS_PATH = (
    "data/processed/walk_forward_predictions.csv"
)

REPORT_PATH = (
    "data/processed/walk_forward_report.csv"
)


# ------------------------------------------------------------
# Features used by the ML model
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "trade_hour",
    "trade_day_of_week",
    "trade_month",
    "trade_quarter",

    "macro_events_1d",
    "macro_events_3d",
    "macro_events_7d",

    "high_impact_events_1d",
    "high_impact_events_3d",
    "high_impact_events_7d",

    "avg_surprise_1d",
    "avg_surprise_3d",
    "avg_surprise_7d",

    "positive_surprises_7d",
    "negative_surprises_7d",

    "cpi_events_7d",
    "fomc_events_7d",
    "employment_events_7d",

    "latest_macro_surprise",
    "days_since_macro_event",

    "fast_window",
    "slow_window",
    "position",
]


def load_dataset():
    """Load and prepare the feature dataset."""

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["entry_time"]
    )

    df = df.sort_values(
        "entry_time"
    ).reset_index(drop=True)

    return df


def calculate_sharpe(returns):
    """Calculate annualized Sharpe ratio."""

    returns = pd.Series(returns).dropna()

    if len(returns) < 2:
        return 0.0

    std = returns.std()

    if std == 0 or pd.isna(std):
        return 0.0

    return (
        returns.mean() / std
    ) * np.sqrt(252)


def calculate_sortino(returns):
    """Calculate annualized Sortino ratio."""

    returns = pd.Series(returns).dropna()

    if len(returns) < 2:
        return 0.0

    downside = returns[returns < 0]

    if len(downside) == 0:
        return 0.0

    downside_std = downside.std()

    if downside_std == 0 or pd.isna(downside_std):
        return 0.0

    return (
        returns.mean() / downside_std
    ) * np.sqrt(252)


def calculate_max_drawdown(pnl):
    """Calculate maximum cumulative P&L drawdown."""

    pnl = pd.Series(pnl).fillna(0)

    cumulative = pnl.cumsum()

    running_max = cumulative.cummax()

    drawdown = (
        cumulative - running_max
    )

    return drawdown.min()


def walk_forward_strategy(
    strategy_id,
    strategy_df,
    train_size=0.60,
    test_size=0.10,
):
    """
    Perform chronological walk-forward validation.

    Example:

    60% train
    10% test
    move forward
    60% train
    10% test
    ...
    """

    strategy_df = (
        strategy_df
        .sort_values("entry_time")
        .reset_index(drop=True)
    )

    n = len(strategy_df)

    initial_train = max(
        30,
        int(n * train_size)
    )

    test_length = max(
        10,
        int(n * test_size)
    )

    predictions = []

    train_end = initial_train

    fold = 1

    while train_end < n:

        test_end = min(
            train_end + test_length,
            n
        )

        train = strategy_df.iloc[
            :train_end
        ].copy()

        test = strategy_df.iloc[
            train_end:test_end
        ].copy()

        if len(test) == 0:
            break

        X_train = train[
            FEATURE_COLUMNS
        ]

        y_train_pnl = train[
            "target_pnl"
        ]

        y_train_win = train[
            "target_win"
        ]

        X_test = test[
            FEATURE_COLUMNS
        ]

        # ----------------------------------------------------
        # Regression model
        # ----------------------------------------------------

        regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

        regressor.fit(
            X_train,
            y_train_pnl
        )

        predicted_pnl = (
            regressor.predict(X_test)
        )

        # ----------------------------------------------------
        # Classification model
        # ----------------------------------------------------

        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        # If training fold contains only one class,
        # classification cannot be trained.
        if y_train_win.nunique() >= 2:

            classifier.fit(
                X_train,
                y_train_win
            )

            predicted_win = (
                classifier.predict(X_test)
            )

        else:

            predicted_win = np.zeros(
                len(X_test),
                dtype=int
            )

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        fold_result = test[
            [
                "trade_id",
                "strategy_id",
                "entry_time",
                "target_pnl",
                "target_return_pct",
                "target_win",
            ]
        ].copy()

        fold_result[
            "predicted_pnl"
        ] = predicted_pnl

        fold_result[
            "predicted_win"
        ] = predicted_win

        fold_result[
            "fold"
        ] = fold

        predictions.append(
            fold_result
        )

        print(
            f"  Fold {fold}: "
            f"train={len(train)}, "
            f"test={len(test)}"
        )

        train_end = test_end

        fold += 1

    if not predictions:
        return pd.DataFrame()

    return pd.concat(
        predictions,
        ignore_index=True
    )


def evaluate_predictions(predictions):
    """Calculate ML and trading metrics."""

    if predictions.empty:
        return {}

    y_true_pnl = predictions[
        "target_pnl"
    ]

    y_pred_pnl = predictions[
        "predicted_pnl"
    ]

    y_true_win = predictions[
        "target_win"
    ]

    y_pred_win = predictions[
        "predicted_win"
    ]

    # --------------------------------------------------------
    # Regression metrics
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_true_pnl,
        y_pred_pnl
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true_pnl,
            y_pred_pnl
        )
    )

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true_win,
        y_pred_win
    )

    precision = precision_score(
        y_true_win,
        y_pred_win,
        zero_division=0
    )

    recall = recall_score(
        y_true_win,
        y_pred_win,
        zero_division=0
    )

    f1 = f1_score(
        y_true_win,
        y_pred_win,
        zero_division=0
    )

    # --------------------------------------------------------
    # Trading strategy based on model prediction
    #
    # Only take trades where predicted P&L > 0.
    # --------------------------------------------------------

    selected = predictions[
        predictions["predicted_pnl"] > 0
    ].copy()

    if selected.empty:

        strategy_pnl = 0.0
        sharpe = 0.0
        sortino = 0.0
        max_drawdown = 0.0

    else:

        strategy_pnl = (
            selected["target_pnl"].sum()
        )

        strategy_returns = (
            selected["target_return_pct"] / 100.0
        )

        sharpe = calculate_sharpe(
            strategy_returns
        )

        sortino = calculate_sortino(
            strategy_returns
        )

        max_drawdown = calculate_max_drawdown(
            selected["target_pnl"]
        )

    return {
        "test_trades": len(predictions),

        "selected_trades": len(selected),

        "MAE": mae,

        "RMSE": rmse,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "strategy_pnl": strategy_pnl,

        "sharpe": sharpe,

        "sortino": sortino,

        "max_drawdown": max_drawdown,
    }


def main():

    print(
        "========================================"
    )
    print(
        "       WALK-FORWARD ML VALIDATION"
    )
    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = load_dataset()

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Strategies: "
        f"{df['strategy_id'].unique()}"
    )

    # --------------------------------------------------------
    # Check features
    # --------------------------------------------------------

    missing_features = [
        col
        for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Run each strategy separately
    # --------------------------------------------------------

    all_predictions = []

    reports = []

    for strategy_id in sorted(
        df["strategy_id"].unique()
    ):

        print("\n----------------------------------------")

        print(
            f"Strategy: {strategy_id}"
        )

        strategy_df = df[
            df["strategy_id"] == strategy_id
        ].copy()

        print(
            f"Trades: {len(strategy_df)}"
        )

        predictions = walk_forward_strategy(
            strategy_id,
            strategy_df
        )

        if predictions.empty:

            print(
                "No predictions generated."
            )

            continue

        predictions[
            "strategy_id"
        ] = strategy_id

        all_predictions.append(
            predictions
        )

        metrics = evaluate_predictions(
            predictions
        )

        metrics[
            "strategy_id"
        ] = strategy_id

        metrics[
            "total_trades"
        ] = len(strategy_df)

        reports.append(
            metrics
        )

        print("\nResults:")

        for key, value in metrics.items():

            if key == "strategy_id":
                continue

            if isinstance(value, float):
                print(
                    f"  {key}: "
                    f"{value:.4f}"
                )
            else:
                print(
                    f"  {key}: {value}"
                )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    if not all_predictions:

        raise RuntimeError(
            "No walk-forward predictions generated."
        )

    final_predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    os.makedirs(
        os.path.dirname(
            PREDICTIONS_PATH
        ),
        exist_ok=True
    )

    final_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report_df = pd.DataFrame(
        reports
    )

    report_df = report_df[
        [
            "strategy_id",
            "total_trades",
            "test_trades",
            "selected_trades",
            "MAE",
            "RMSE",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "strategy_pnl",
            "sharpe",
            "sortino",
            "max_drawdown",
        ]
    ]

    report_df.to_csv(
        REPORT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n========================================")
    print(
        "       WALK-FORWARD COMPLETE"
    )
    print("========================================")

    print(
        f"Predictions: "
        f"{PREDICTIONS_PATH}"
    )

    print(
        f"Report: "
        f"{REPORT_PATH}"
    )

    print("\nFinal strategy comparison:")

    print(
        report_df.to_string(
            index=False
        )
    )

    print("========================================")


if __name__ == "__main__":
    main()

