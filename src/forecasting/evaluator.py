"""
EVALUATOR.PY — The Judge
==========================

WHAT THIS FILE DOES:
    Compares ALL models (XGBoost, ARIMA, naive baseline) side by side
    using multiple metrics, so you can see which model is best and WHY.

THE THREE METRICS EXPLAINED (like you're 10):

    MAE (Mean Absolute Error):
        "On average, how many units off are my predictions?"
        If MAE = 3, your predictions are wrong by about 3 units on average.
        LOWER IS BETTER.

    RMSE (Root Mean Squared Error):
        Same as MAE but PUNISHES big errors more.
        If you're off by 1 unit 99 times and off by 100 units once,
        MAE says ~2 (not bad!), RMSE says ~10 (that's bad!).
        RMSE catches the ONE time you're way off.
        LOWER IS BETTER.

    MAPE (Mean Absolute Percentage Error):
        "On average, what PERCENTAGE off are my predictions?"
        If MAPE = 8%, your predictions are ~8% wrong on average.
        This is scale-independent: 8% means the same whether
        you're predicting 10 units or 10,000 units.
        LOWER IS BETTER. Under 10% is good. Under 8% is excellent.

WHY WE COMPARE AGAINST A NAIVE BASELINE:
    The naive baseline predicts: "tomorrow = today" (just repeat the last value).
    If your fancy ML model can't beat this, it's useless.

    It's like saying "my new GPS app is amazing!" — but if walking
    in a straight line gets you there faster, your app is broken.

INTERVIEW TIP:
    "I evaluate forecasting models using MAE, RMSE, and MAPE against
    a naive persistence baseline. XGBoost achieved X% improvement in
    MAE over the baseline, with feature importance analysis confirming
    that temporal lag features and rolling statistics were the strongest
    predictors — consistent with domain knowledge that recent demand
    is the best predictor of near-term demand."
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_metrics(actuals: np.ndarray, predictions: np.ndarray, model_name: str) -> dict:
    """
    Calculate MAE, RMSE, and MAPE for a set of predictions.

    PARAMETERS:
        actuals     : the real values (what actually happened)
        predictions : what the model predicted
        model_name  : name for display ("XGBoost", "ARIMA", etc.)

    RETURNS:
        Dictionary with all metrics
    """
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    # MAPE — only on non-zero actuals (can't divide by zero)
    mask = actuals > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
    else:
        mape = 0.0

    return {
        "model": model_name,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 1),
    }


def run_comparison():
    """
    Load predictions from both models and compare them side by side.
    """
    print("=" * 60)
    print("Model comparison: XGBoost vs ARIMA vs Naive baseline")
    print("=" * 60)

    # Load predictions
    xgb_results = pd.read_csv("data/processed/xgboost_predictions.csv")
    arima_results = pd.read_csv("data/processed/arima_predictions.csv")

    actuals = xgb_results["actual"].values

    # Calculate metrics for each model
    xgb_metrics = calculate_metrics(actuals, xgb_results["predicted"].values, "XGBoost")
    arima_metrics = calculate_metrics(actuals, arima_results["predicted"].values, "ARIMA")
    naive_metrics = calculate_metrics(actuals, xgb_results["naive"].values, "Naive (lag_1)")

    all_metrics = [xgb_metrics, arima_metrics, naive_metrics]

    # Print comparison table
    print(f"\n{'Model':<20} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print("-" * 46)
    for m in all_metrics:
        mape_str = f"{m['mape']}%"
        print(f"{m['model']:<20} {m['mae']:>8.2f} {m['rmse']:>8.2f} {mape_str:>8}")

    # Find the best model
    best_mae = min(all_metrics, key=lambda x: x["mae"])
    best_mape = min(all_metrics, key=lambda x: x["mape"])

    print(f"\n{'=' * 40}")
    print(f"WINNER")
    print(f"{'=' * 40}")
    print(f"  Best MAE:  {best_mae['model']} ({best_mae['mae']} units)")
    print(f"  Best MAPE: {best_mape['model']} ({best_mape['mape']}%)")

    # Improvement over baseline
    naive_mae = naive_metrics["mae"]
    naive_mape = naive_metrics["mape"]

    print(f"\n  Improvement over naive baseline:")
    for m in [xgb_metrics, arima_metrics]:
        mae_imp = (1 - m["mae"] / naive_mae) * 100 if naive_mae > 0 else 0
        mape_imp = (1 - m["mape"] / naive_mape) * 100 if naive_mape > 0 else 0
        print(f"    {m['model']}: MAE {mae_imp:+.1f}%, MAPE {mape_imp:+.1f}%")

    # Per-part analysis
    print(f"\n{'=' * 40}")
    print(f"PER-PART ANALYSIS (what interviewers love)")
    print(f"{'=' * 40}")

    features_df = pd.read_csv("data/processed/ml_features.csv")
    features_df = features_df.sort_values("transaction_date").reset_index(drop=True)
    split_index = int(len(features_df) * 0.8)
    test_features = features_df.iloc[split_index:].reset_index(drop=True)

    if "part_number" in test_features.columns and len(test_features) == len(actuals):
        xgb_preds = xgb_results["predicted"].values

        for part in test_features["part_number"].unique():
            part_mask = test_features["part_number"] == part
            part_actuals = actuals[part_mask]
            part_preds = xgb_preds[part_mask]

            if len(part_actuals) > 0:
                part_mae = mean_absolute_error(part_actuals, part_preds)
                non_zero = part_actuals > 0
                if non_zero.sum() > 0:
                    part_mape = np.mean(np.abs((part_actuals[non_zero] - part_preds[non_zero]) / part_actuals[non_zero])) * 100
                else:
                    part_mape = 0.0
                part_type = test_features[part_mask]["demand_type"].iloc[0] if "demand_type" in test_features.columns else "unknown"
                print(f"  {part:<26} type={part_type:<18} MAE={part_mae:.2f}  MAPE={part_mape:.1f}%")

    # Save comparison results
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.to_csv("data/processed/model_comparison.csv", index=False)
    print(f"\n  Comparison saved to data/processed/model_comparison.csv")

    return all_metrics


if __name__ == "__main__":
    metrics = run_comparison()
