"""
ARIMA_MODEL.PY — The Time-Series Specialist
==============================================

WHAT IS ARIMA? 

    Imagine tracking your height every month. You notice:
    1. You're generally getting TALLER (that's the TREND)
    2. You grow faster in summer than winter (that's SEASONALITY)
    3. Some months you grow a tiny bit more or less than expected (that's NOISE)

    ARIMA separates these three patterns and uses them to predict
    your height 6 months from now.

    The name ARIMA stands for:
    - AR (Auto-Regressive): "My height this month depends on my height last month"
    - I  (Integrated): "Let me remove the growth trend to see the seasonal pattern"
    - MA (Moving Average): "Let me smooth out the random noise"

WHY ARIMA ALONGSIDE XGBOOST:
    XGBoost sees ALL features at once but treats each row independently.
    ARIMA ONLY looks at the time dimension but understands the SEQUENCE.

    XGBoost:  "Given price=10, region=Asia, day=Monday → predict 42"
    ARIMA:    "Demand was 40, 41, 43, 42 → predict ~43 (upward trend)"

    They complement each other like a generalist and a specialist.

BUSINESS PERSPECTIVE:
    ARIMA answers: "Based on the historical demand pattern for this specific
    part, what will demand be next week/month/quarter?"

    It's especially good for:
    - Seasonal products (Q4 spikes for electronics)
    - Trending products (gradually increasing/decreasing demand)
    - Stable products (steady demand with small fluctuations)

SIMPLIFIED APPROACH:
    Full ARIMA requires the statsmodels library. We implement a 
    SIMPLIFIED version using exponential smoothing
    and basic time-series decomposition. This teaches the same concepts
    and is actually what many production systems use.

INTERVIEW TIP:
    "I used a dual-model approach: XGBoost for cross-feature pattern
    recognition and an exponential smoothing model for time-series
    decomposition. The ARIMA-style model captured weekly seasonality
    and short-term trends, complementing XGBoost's ability to leverage
    external features like price and NPI signals."
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os


class SimpleTimeSeriesModel:
    """
    A simplified time-series model that captures the key ideas behind ARIMA:
    - Trend (is demand going up or down?)
    - Seasonality (does demand repeat weekly?)
    - Smoothing (what's the underlying demand level?)

    This uses EXPONENTIAL SMOOTHING — a technique where recent observations
    get more weight than older ones.

    WHAT IS EXPONENTIAL SMOOTHING? 
        Imagine you're guessing tomorrow's temperature.
        - Simple approach: average of ALL past days (gives equal weight)
        - Exponential smoothing: recent days matter MORE than old days

        If alpha=0.3:
          prediction = 0.3 * today + 0.21 * yesterday + 0.147 * two_days_ago + ...

        The weights decay exponentially (0.3, 0.21, 0.147, 0.103, ...).
        Recent data has the most influence.

    PARAMETERS:
        alpha (0.3): Smoothing factor for LEVEL (0=ignore new data, 1=only use latest)
        beta (0.1):  Smoothing factor for TREND (how quickly trend changes)
        period (7):  Seasonality period (7=weekly pattern)
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1, period: int = 7):
        self.alpha = alpha      # Level smoothing
        self.beta = beta        # Trend smoothing
        self.period = period    # Seasonality period (7 = weekly)
        self.models = {}        # One model per part_number
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        """
        Train a separate time-series model for each part_number.

        WHY PER-PART MODELS?
            Each part has its own demand pattern. AMAT-RF-7800-GEN (RF generator)
            has completely different seasonality than AMAT-CMP-4100-PAD (polishing pad).
            Training one model per part captures these individual patterns.

        WHAT WE LEARN FOR EACH PART:
            1. Level (l): The current demand baseline
            2. Trend (b): Is demand going up or down, and by how much?
            3. Seasonal factors (s): How does each day of the week differ?
        """
        df = df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df = df.sort_values(["part_number", "transaction_date"])

        print(f"\nTraining time-series models...")
        print(f"  Alpha (level smoothing): {self.alpha}")
        print(f"  Beta (trend smoothing):  {self.beta}")
        print(f"  Period (seasonality):    {self.period} days")

        parts = df["part_number"].unique()
        print(f"  Training {len(parts)} individual models (one per part)...")

        for part in parts:
            part_data = df[df["part_number"] == part]["demand_quantity"].values

            if len(part_data) < self.period + 2:
                # Not enough data for this part — use simple mean
                self.models[part] = {
                    "type": "mean",
                    "mean": np.mean(part_data),
                    "last_values": part_data[-self.period:].tolist() if len(part_data) >= self.period else part_data.tolist(),
                }
                continue

            # Initialize level and trend
            level = np.mean(part_data[:self.period])
            trend = (np.mean(part_data[self.period:2*self.period]) - level) / self.period if len(part_data) >= 2 * self.period else 0

            # Initialize seasonal factors
            # Each day-of-week gets a factor: >1 means above average, <1 means below
            seasonal = np.ones(self.period)
            for i in range(min(self.period, len(part_data))):
                if level > 0:
                    seasonal[i] = part_data[i] / level

            # Update through the series using exponential smoothing
            for t in range(self.period, len(part_data)):
                season_index = t % self.period
                observed = part_data[t]

                # Prevent division by zero
                seasonal_factor = max(seasonal[season_index], 0.01)

                # Update level (deseasonalized)
                new_level = self.alpha * (observed / seasonal_factor) + (1 - self.alpha) * (level + trend)

                # Update trend
                new_trend = self.beta * (new_level - level) + (1 - self.beta) * trend

                # Update seasonal factor
                if new_level > 0:
                    seasonal[season_index] = self.alpha * (observed / new_level) + (1 - self.alpha) * seasonal[season_index]

                level = new_level
                trend = new_trend

            self.models[part] = {
                "type": "holt_winters",
                "level": level,
                "trend": trend,
                "seasonal": seasonal.tolist(),
                "last_values": part_data[-self.period:].tolist(),
            }

        self.is_trained = True
        print(f"  Training complete! {len(self.models)} models created.")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for each row in the dataframe.

        For each row, we:
        1. Look up the part's model
        2. Use level + trend + seasonal factor to predict
        3. Clip to non-negative (demand can't be negative)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")

        df = df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        predictions = []

        for _, row in df.iterrows():
            part = row.get("part_number", "")

            if part not in self.models:
                predictions.append(0)
                continue

            model = self.models[part]

            if model["type"] == "mean":
                predictions.append(model["mean"])
            else:
                # Holt-Winters prediction
                level = model["level"]
                trend = model["trend"]
                seasonal = model["seasonal"]

                # Which season index are we?
                date = row["transaction_date"]
                season_index = date.dayofweek % len(seasonal)

                # Prediction = (level + trend * steps_ahead) * seasonal_factor
                prediction = (level + trend) * seasonal[season_index]
                predictions.append(max(prediction, 0))

        return np.array(predictions)

    def save(self, path: str = "models/arima_model.pkl"):
        """Save all trained models to disk."""
        import pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models,
                "params": {"alpha": self.alpha, "beta": self.beta, "period": self.period},
            }, f)
        print(f"  Model saved to {path}")


def train_and_evaluate():
    """
    Run the complete ARIMA-style training and evaluation pipeline.
    """
    print("=" * 60)
    print("ARIMA-Style Time-Series Forecasting")
    print("=" * 60)

    # Load data
    df = pd.read_csv("data/processed/ml_features.csv")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df = df.sort_values("transaction_date").reset_index(drop=True)

    # Time-based split (same as XGBoost for fair comparison)
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    print(f"\nData split:")
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")

    # Train
    model = SimpleTimeSeriesModel(alpha=0.3, beta=0.1, period=7)
    model.train(train_df)

    # Predict
    print(f"\nMaking predictions on test set...")
    predictions = model.predict(test_df)
    actuals = test_df["demand_quantity"].values

    # Evaluate
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    mask = actuals > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
    else:
        mape = 0.0

    # Naive baseline
    if "lag_1" in test_df.columns:
        naive_predictions = test_df["lag_1"].values
    else:
        naive_predictions = np.roll(actuals, 1)
        naive_predictions[0] = actuals[0]

    naive_mae = mean_absolute_error(actuals, naive_predictions)
    naive_mask = actuals > 0
    if naive_mask.sum() > 0:
        naive_mape = np.mean(np.abs((actuals[naive_mask] - naive_predictions[naive_mask]) / actuals[naive_mask])) * 100
    else:
        naive_mape = 0.0

    print(f"\n{'=' * 40}")
    print(f"ARIMA-STYLE RESULTS")
    print(f"{'=' * 40}")
    print(f"  MAE:     {mae:>10.2f} units")
    print(f"  RMSE:    {rmse:>10.2f} units")
    print(f"  MAPE:    {mape:>9.1f}%")
    print(f"\n  NAIVE BASELINE:")
    print(f"  MAE:     {naive_mae:>10.2f} units")
    print(f"  MAPE:    {naive_mape:>9.1f}%")

    # Save
    model.save()

    results_df = pd.DataFrame({
        "actual": actuals,
        "predicted": predictions,
    })
    results_df.to_csv("data/processed/arima_predictions.csv", index=False)
    print(f"\n  Predictions saved to data/processed/arima_predictions.csv")

    return model, results_df


if __name__ == "__main__":
    model, results = train_and_evaluate()
