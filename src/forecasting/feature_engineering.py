"""
FEATURE_ENGINEERING.PY — The Study Material Creator
=====================================================

WHAT THIS FILE DOES:
    Takes our clean validated data and creates NEW columns (features)
    that help the ML model make better predictions.

    The raw data says: "Part X sold 5 units on Jan 1st."
    After feature engineering, it says: "Part X sold 5 units on Jan 1st,
    which was a Thursday, in week 1, in Q1. Yesterday it sold 3 units.
    The average over the last 7 days was 4.2 units. Last week same day
    it sold 6 units."

    The ML model needs these patterns to make predictions.

WHY THIS IS THE MOST IMPORTANT FILE IN PHASE 2:
    There's a saying in data science: "A simple model with great features
    beats a complex model with bad features." Feature engineering is where
    80% of the predictive power comes from. The model algorithm (XGBoost,
    ARIMA) is just the engine — features are the fuel.

BUSINESS PERSPECTIVE:
    A supply chain manager thinks in patterns:
    - "Demand is always higher on Mondays" (day-of-week feature)
    - "Q4 is our busy season" (quarter feature)
    - "If demand was high yesterday, it's probably high today" (lag feature)
    - "Demand has been trending up over the last month" (rolling mean feature)

    We're encoding this human intuition into numbers that the model can learn.

INTERVIEW TIP:
    "I engineered 15+ temporal and statistical features including lag
    variables, rolling window aggregates, and calendar decompositions.
    Feature importance analysis showed that 7-day rolling mean and
    1-day lag were the strongest predictors, which aligns with supply
    chain domain knowledge — recent demand is the best predictor of
    near-term demand."
"""

import pandas as pd
import numpy as np


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar-based features from the transaction_date column.

    WHY EACH FEATURE EXISTS:

    day_of_week (0=Monday, 6=Sunday):
        BUSINESS: Factories order more on weekdays than weekends.
        TECHNICAL: Captures weekly cyclical patterns.

    day_of_month (1-31):
        BUSINESS: Some companies place orders at month start/end.
        TECHNICAL: Captures monthly cyclical patterns.

    week_of_year (1-52):
        BUSINESS: Q4 (weeks 40-52) is busy season for electronics.
        TECHNICAL: Captures annual seasonality.

    month (1-12):
        BUSINESS: December = holiday demand spike. February = slow.
        TECHNICAL: Coarser seasonality than week_of_year.

    quarter (1-4):
        BUSINESS: Budget cycles run quarterly. Q4 = spending spree.
        TECHNICAL: Captures fiscal quarter patterns.

    is_weekend (0 or 1):
        BUSINESS: Weekend demand is structurally different.
        TECHNICAL: Binary flag — simpler than day_of_week for some models.

    is_month_start / is_month_end (0 or 1):
        BUSINESS: Companies rush orders at month end to hit targets.
        TECHNICAL: Captures edge-of-period behavior.
    """
    df = df.copy()

    # Convert string dates to datetime objects
    # pandas needs this to extract day, month, year etc.
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # Extract components
    df["day_of_week"] = df["transaction_date"].dt.dayofweek       # 0=Mon, 6=Sun
    df["day_of_month"] = df["transaction_date"].dt.day             # 1-31
    df["week_of_year"] = df["transaction_date"].dt.isocalendar().week.astype(int)  # 1-52
    df["month"] = df["transaction_date"].dt.month                  # 1-12
    df["quarter"] = df["transaction_date"].dt.quarter              # 1-4
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)        # 1 if Sat/Sun
    df["is_month_start"] = df["transaction_date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["transaction_date"].dt.is_month_end.astype(int)

    return df


def create_lag_features(df: pd.DataFrame, lags: list[int] = None) -> pd.DataFrame:
    """
    Create LAG features — "what was demand N days ago?"

    WHAT IS A LAG FEATURE?
        If today is January 10th and demand was 42 units:
        - lag_1 = demand on Jan 9th (yesterday)
        - lag_7 = demand on Jan 3rd (same day last week)
        - lag_14 = demand on Dec 27th (two weeks ago)

    WHY LAGS MATTER:
        BUSINESS: "If we sold 100 units yesterday, we'll probably sell
        close to 100 today." Recent demand is the single best predictor
        of near-term demand. This is called "autocorrelation."

        TECHNICAL: Lag features turn a time-series problem into a
        standard tabular ML problem. Without lags, XGBoost can't "see"
        what happened yesterday — it only sees today's row.

    WHY THESE SPECIFIC LAGS (1, 3, 7, 14):
        lag_1  → Yesterday. Captures day-to-day momentum.
        lag_3  → Three days ago. Captures short-term trends.
        lag_7  → Same day last week. Captures weekly patterns.
        lag_14 → Two weeks ago. Captures biweekly patterns.

    IMPORTANT: Lags create NaN values at the start of each group.
    For lag_7, the first 7 rows won't have a value (there's no
    "7 days ago" for the first week). We handle this later.
    """
    df = df.copy()

    if lags is None:
        lags = [1, 3, 7, 14]

    # Sort by part + date so lags are computed in chronological order
    df = df.sort_values(["part_number", "transaction_date"]).reset_index(drop=True)

    for lag in lags:
        # shift(N) takes the value from N rows earlier
        # We group by part_number so Part A's lag doesn't leak into Part B's data
        df[f"lag_{lag}"] = df.groupby("part_number")["demand_quantity"].shift(lag)

    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ROLLING WINDOW features — averages and statistics over recent periods.

    WHAT IS A ROLLING MEAN?
        "What was the average demand over the last 7 days?"
        On Jan 10th, the 7-day rolling mean looks at Jan 4-10 and averages them.

    WHY ROLLING FEATURES MATTER:
        BUSINESS: A supply chain manager doesn't just look at yesterday —
        they look at "how has this week been?" and "how has this month been?"
        Rolling means smooth out random noise and reveal the underlying trend.

        TECHNICAL: Individual days are noisy (random spikes/dips).
        Rolling means provide a smoothed signal the model can learn from.
        They also capture trend direction — if the 7-day mean is higher than
        the 14-day mean, demand is trending UP.

    FEATURES CREATED:
        rolling_mean_7   → Average demand over last 7 days (weekly trend)
        rolling_mean_14  → Average demand over last 14 days (biweekly trend)
        rolling_std_7    → Standard deviation over last 7 days (volatility)
        rolling_min_7    → Minimum demand in last 7 days (floor level)
        rolling_max_7    → Maximum demand in last 7 days (ceiling level)

    INTERVIEW TIP:
        "Rolling standard deviation captures demand volatility — a part with
        std=2 has stable demand while std=20 is highly volatile. This helps
        the model adjust confidence intervals and safety stock recommendations."
    """
    df = df.copy()
    df = df.sort_values(["part_number", "transaction_date"]).reset_index(drop=True)

    # Group by part so each part's rolling stats are independent
    grouped = df.groupby("part_number")["demand_quantity"]

    # Rolling mean: average over the window
    # min_periods=1 means "compute even if we don't have the full window yet"
    # shift(1) prevents data leakage — we don't include TODAY's value
    df["rolling_mean_7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df["rolling_mean_14"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=14, min_periods=1).mean()
    )

    # Rolling standard deviation: how much does demand fluctuate?
    df["rolling_std_7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).std()
    )

    # Rolling min and max: range of recent demand
    df["rolling_min_7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).min()
    )
    df["rolling_max_7"] = grouped.transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).max()
    )

    return df


def create_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns (text) as numbers.

    WHY: ML models can't read text. "NPI" and "North America" mean
    nothing to XGBoost. We need to convert them to numbers.

    TWO APPROACHES:
        1. Label encoding: NPI=0, standard=1, service_campaign=2, reliability=3
           Simple but implies an order (3 > 2 > 1) which isn't true.

        2. One-hot encoding: Create separate columns for each category.
           NPI → [1,0,0,0], standard → [0,1,0,0], etc.
           No implied ordering. Better for tree-based models.

    We use LABEL ENCODING here because:
    - XGBoost handles label-encoded categories natively
    - One-hot encoding with many categories creates too many columns
    - For tree-based models, label encoding works well in practice
    """
    df = df.copy()

    # Create mappings
    demand_type_map = {"NPI": 0, "standard": 1, "service_campaign": 2, "reliability": 3}
    region_map = {"North America": 0, "Asia Pacific": 1, "Europe": 2}

    df["demand_type_encoded"] = df["demand_type"].map(demand_type_map).fillna(-1).astype(int)
    df["region_encoded"] = df["region"].map(region_map).fillna(-1).astype(int)

    # Has NPI signal (already exists from Phase 1, but ensure it's numeric)
    if "has_npi_signal" not in df.columns:
        df["has_npi_signal"] = df["npi_signal"].notna().astype(int)

    return df


def prepare_ml_dataset(
    input_path: str = "data/processed/m5_amat_validated.csv",
    output_path: str = "data/processed/ml_features.csv",
) -> pd.DataFrame:
    """
    THE MAIN FUNCTION: Run all feature engineering steps and produce
    a dataset ready for ML model training.

    PIPELINE:
        1. Load validated data from Phase 1
        2. Create date features (day, month, quarter, etc.)
        3. Create lag features (yesterday, last week, etc.)
        4. Create rolling features (7-day mean, 14-day mean, etc.)
        5. Encode categorical columns as numbers
        6. Handle missing values (NaN from lags/rolling)
        7. Save the result

    RETURNS:
        DataFrame ready for train/test split and model training
    """
    import os

    print("=" * 60)
    print("Feature engineering for ML forecasting")
    print("=" * 60)

    # Step 1: Load
    df = pd.read_csv(input_path)
    print(f"\nLoaded {len(df):,} records from {input_path}")
    print(f"  Columns: {len(df.columns)}")

    # Step 2: Date features
    print("\nCreating date features...")
    df = create_date_features(df)
    print(f"  Added: day_of_week, day_of_month, week_of_year, month, quarter, is_weekend")

    # Step 3: Lag features
    print("Creating lag features...")
    df = create_lag_features(df, lags=[1, 3, 7, 14])
    print(f"  Added: lag_1, lag_3, lag_7, lag_14")

    # Step 4: Rolling features
    print("Creating rolling window features...")
    df = create_rolling_features(df)
    print(f"  Added: rolling_mean_7, rolling_mean_14, rolling_std_7, rolling_min_7, rolling_max_7")

    # Step 5: Category encoding
    print("Encoding categorical features...")
    df = create_category_features(df)
    print(f"  Added: demand_type_encoded, region_encoded")

    # Step 6: Handle missing values
    # Lag and rolling features create NaN at the start of each group.
    # We drop these rows because the model can't learn from incomplete data.
    rows_before = len(df)
    df = df.dropna(subset=["lag_1"])  # Drop rows where even lag_1 is NaN
    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    print(f"\nDropped {rows_dropped} rows with NaN lag values (first day per part)")

    # Fill remaining NaN in rolling features with 0
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    df[rolling_cols] = df[rolling_cols].fillna(0)

    # Fill NaN in lag columns (lag_14 might have NaN even when lag_1 doesn't)
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    df[lag_cols] = df[lag_cols].fillna(0)

    # Step 7: Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Summary
    feature_cols = [
        "day_of_week", "day_of_month", "week_of_year", "month", "quarter",
        "is_weekend", "is_month_start", "is_month_end",
        "lag_1", "lag_3", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
        "rolling_min_7", "rolling_max_7",
        "demand_type_encoded", "region_encoded", "has_npi_signal",
        "unit_cost",
    ]
    existing_features = [c for c in feature_cols if c in df.columns]

    print(f"\n{'=' * 40}")
    print(f"FEATURE ENGINEERING SUMMARY")
    print(f"{'=' * 40}")
    print(f"  Total records:     {len(df):>8,}")
    print(f"  Total columns:     {len(df.columns):>8}")
    print(f"  ML features:       {len(existing_features):>8}")
    print(f"  Target variable:   demand_quantity")
    print(f"\n  Feature list:")
    for f in existing_features:
        print(f"    - {f}")

    return df


# Define which columns the model should use
FEATURE_COLUMNS = [
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter",
    "is_weekend", "is_month_start", "is_month_end",
    "lag_1", "lag_3", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "rolling_min_7", "rolling_max_7",
    "demand_type_encoded", "region_encoded", "has_npi_signal",
    "unit_cost",
]

TARGET_COLUMN = "demand_quantity"


if __name__ == "__main__":
    df = prepare_ml_dataset()
    print(f"\nFirst 3 rows of features:")
    feature_cols_present = [c for c in FEATURE_COLUMNS if c in df.columns]
    print(df[feature_cols_present].head(3).to_string(index=False))
