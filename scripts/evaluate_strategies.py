
import os
import numpy as np
import pandas as pd


PREDICTIONS_PATH = (
    "data/processed/walk_forward_predictions.csv"
)

OUTPUT_PATH = (
    "data/processed/strategy_evaluation.csv"
)


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

    if len(downside) < 2:
        return 0.0

    downside_std = downside.std()

    if downside_std == 0 or pd.isna(downside_std):
        return 0.0

    return (
        returns.mean() / downside_std
    ) * np.sqrt(252)


def calculate_max_drawdown(pnl):
    """Calculate maximum drawdown from cumulative P&L."""

    pnl = pd.Series(pnl).fillna(0)

    cumulative = pnl.cumsum()

    running_max = cumulative.cummax()

    drawdown = cumulative - running_max

    return drawdown.min()


def calculate_profit_factor(pnl):
    """Calculate gross profit / gross loss."""

    pnl = pd.Series(pnl).dropna()

    gross_profit = pnl[pnl > 0].sum()

    gross_loss = abs(
        pnl[pnl < 0].sum()
    )

    if gross_loss == 0:
        return np.inf

    return gross_profit / gross_loss


def evaluate_strategy(predictions, strategy_id):
    """Evaluate one strategy."""

    df = predictions[
        predictions["strategy_id"] == strategy_id
    ].copy()

    if df.empty:
        return None

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "entry_time"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # BASELINE
    #
    # Take every test trade.
    # --------------------------------------------------------

    baseline_pnl = df["target_pnl"].sum()

    baseline_win_rate = df["target_win"].mean()

    baseline_returns = (
        df["target_return_pct"] / 100.0
    )

    baseline_sharpe = calculate_sharpe(
        baseline_returns
    )

    baseline_sortino = calculate_sortino(
        baseline_returns
    )

    baseline_drawdown = calculate_max_drawdown(
        df["target_pnl"]
    )

    baseline_profit_factor = calculate_profit_factor(
        df["target_pnl"]
    )

    # --------------------------------------------------------
    # ML-FILTERED STRATEGY
    #
    # Only take trades where predicted P&L > 0.
    # --------------------------------------------------------

    selected = df[
        df["predicted_pnl"] > 0
    ].copy()

    selected_trades = len(selected)

    selection_rate = (
        selected_trades / len(df)
        if len(df) > 0
        else 0.0
    )

    if selected.empty:

        ml_pnl = 0.0
        ml_win_rate = 0.0
        ml_sharpe = 0.0
        ml_sortino = 0.0
        ml_drawdown = 0.0
        ml_profit_factor = 0.0
        avg_pnl = 0.0

    else:

        ml_pnl = selected["target_pnl"].sum()

        ml_win_rate = selected["target_win"].mean()

        ml_returns = (
            selected["target_return_pct"] / 100.0
        )

        ml_sharpe = calculate_sharpe(
            ml_returns
        )

        ml_sortino = calculate_sortino(
            ml_returns
        )

        ml_drawdown = calculate_max_drawdown(
            selected["target_pnl"]
        )

        ml_profit_factor = calculate_profit_factor(
            selected["target_pnl"]
        )

        avg_pnl = selected["target_pnl"].mean()

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    pnl_improvement = (
        ml_pnl - baseline_pnl
    )

    sharpe_improvement = (
        ml_sharpe - baseline_sharpe
    )

    return {
        "strategy_id": strategy_id,

        # Test population
        "test_trades": len(df),

        # Baseline metrics
        "baseline_pnl": baseline_pnl,
        "baseline_win_rate": baseline_win_rate,
        "baseline_sharpe": baseline_sharpe,
        "baseline_sortino": baseline_sortino,
        "baseline_max_drawdown": baseline_drawdown,
        "baseline_profit_factor": baseline_profit_factor,

        # ML-filtered metrics
        "selected_trades": selected_trades,
        "selection_rate": selection_rate,
        "ml_pnl": ml_pnl,
        "ml_win_rate": ml_win_rate,
        "ml_sharpe": ml_sharpe,
        "ml_sortino": ml_sortino,
        "ml_max_drawdown": ml_drawdown,
        "ml_profit_factor": ml_profit_factor,
        "avg_pnl_per_selected_trade": avg_pnl,

        # Improvements
        "pnl_improvement": pnl_improvement,
        "sharpe_improvement": sharpe_improvement,
    }


def choose_strategy(report):
    """
    Select preferred strategy.

    Primary criterion:
        ML-filtered Sharpe

    Secondary criterion:
        ML-filtered P&L
    """

    if report.empty:
        return None

    ranked = report.sort_values(
        by=[
            "ml_sharpe",
            "ml_pnl"
        ],
        ascending=[
            False,
            False
        ]
    )

    return ranked.iloc[0]["strategy_id"]


def main():

    print("========================================")
    print("       STRATEGY PERFORMANCE AUDIT")
    print("========================================")

    # --------------------------------------------------------
    # Check input file
    # --------------------------------------------------------

    if not os.path.exists(PREDICTIONS_PATH):

        raise FileNotFoundError(
            f"Missing file: {PREDICTIONS_PATH}"
        )

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    print("\nLoading walk-forward predictions...")

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=["entry_time"]
    )

    print(
        f"Prediction rows: {len(predictions)}"
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "strategy_id",
        "entry_time",
        "target_pnl",
        "target_return_pct",
        "target_win",
        "predicted_pnl",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in predictions.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Evaluate every strategy
    # --------------------------------------------------------

    strategies = sorted(
        predictions["strategy_id"].unique()
    )

    print(
        f"Strategies found: {len(strategies)}"
    )

    results = []

    for strategy_id in strategies:

        print(
            f"\nEvaluating: {strategy_id}"
        )

        result = evaluate_strategy(
            predictions,
            strategy_id
        )

        if result is not None:
            results.append(result)

    if not results:

        raise RuntimeError(
            "No strategy results were generated."
        )

    report = pd.DataFrame(results)

    # --------------------------------------------------------
    # Rank strategies
    # --------------------------------------------------------

    report = report.sort_values(
        by=[
            "ml_sharpe",
            "ml_pnl"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(drop=True)

    report.insert(
        0,
        "rank",
        range(1, len(report) + 1)
    )

    # --------------------------------------------------------
    # Select recommended strategy
    # --------------------------------------------------------

    selected_strategy = choose_strategy(
        report
    )

    report["recommended"] = (
        report["strategy_id"]
        == selected_strategy
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    report.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Print comparison
    # --------------------------------------------------------

    print("\n========================================")
    print("       STRATEGY COMPARISON")
    print("========================================")

    display_columns = [
        "rank",
        "strategy_id",
        "test_trades",
        "baseline_pnl",
        "baseline_win_rate",
        "baseline_sharpe",
        "ml_pnl",
        "ml_win_rate",
        "ml_sharpe",
        "ml_sortino",
        "ml_max_drawdown",
        "ml_profit_factor",
        "selection_rate",
        "pnl_improvement",
    ]

    print(
        report[
            display_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    selected_row = report[
        report["strategy_id"]
        == selected_strategy
    ].iloc[0]

    print("\n========================================")
    print("       RECOMMENDED STRATEGY")
    print("========================================")

    print(
        f"Strategy: {selected_strategy}"
    )

    print(
        f"ML-filtered P&L: "
        f"{selected_row['ml_pnl']:.2f}"
    )

    print(
        f"ML-filtered Sharpe: "
        f"{selected_row['ml_sharpe']:.4f}"
    )

    print(
        f"ML-filtered Sortino: "
        f"{selected_row['ml_sortino']:.4f}"
    )

    print(
        f"Maximum Drawdown: "
        f"{selected_row['ml_max_drawdown']:.2f}"
    )

    print(
        f"Win Rate: "
        f"{selected_row['ml_win_rate']:.2%}"
    )

    print(
        f"Selected Trade Rate: "
        f"{selected_row['selection_rate']:.2%}"
    )

    print(
        f"Profit Factor: "
        f"{selected_row['ml_profit_factor']:.4f}"
    )

    print("\nReport saved to:")

    print(
        OUTPUT_PATH
    )

    print("========================================")


if __name__ == "__main__":
    main()