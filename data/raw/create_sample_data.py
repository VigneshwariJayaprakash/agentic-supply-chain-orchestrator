"""
CREATE SAMPLE M5-STYLE DATA
============================
This script creates small sample CSV files that mimic the real M5 dataset.
The real M5 dataset has 46 million+ rows — way too big to include in a repo.
We create ~1,000 rows so you can develop and test locally.

WHY: You always want sample data for development. You don't want to wait
10 minutes to load 46M rows every time you test a one-line code change.
The real dataset gets downloaded separately from Kaggle.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)  # Makes results reproducible — same "random" numbers every time

# ─── CONFIG ────────────────────────────────────────────────────────────────
# These mirror the real M5 dataset structure exactly
ITEMS = [
    "HOBBIES_1_001", "HOBBIES_1_008", "HOBBIES_2_004",   # → NPI demand
    "HOUSEHOLD_1_012", "HOUSEHOLD_2_067",                   # → Standard demand
    "FOODS_1_001", "FOODS_2_099",                           # → Service campaign
    "FOODS_3_042",                                          # → Reliability
]

STORES = ["CA_1", "CA_2", "TX_1", "TX_3", "WI_1", "WI_3"]

DEPARTMENTS = {
    "HOBBIES_1_001": "HOBBIES_1", "HOBBIES_1_008": "HOBBIES_1",
    "HOBBIES_2_004": "HOBBIES_2",
    "HOUSEHOLD_1_012": "HOUSEHOLD_1", "HOUSEHOLD_2_067": "HOUSEHOLD_2",
    "FOODS_1_001": "FOODS_1", "FOODS_2_099": "FOODS_2",
    "FOODS_3_042": "FOODS_3",
}

CATEGORIES = {
    "HOBBIES_1": "HOBBIES", "HOBBIES_2": "HOBBIES",
    "HOUSEHOLD_1": "HOUSEHOLD", "HOUSEHOLD_2": "HOUSEHOLD",
    "FOODS_1": "FOODS", "FOODS_2": "FOODS", "FOODS_3": "FOODS",
}

EVENTS = [None, None, None, None, None, "SuperBowl", "NPI_Launch_Q1",
           "ServiceCampaign_A", None, None]  # Most days have no event


def create_sales_data():
    """
    Create sales_train_evaluation.csv — the main dataset.

    In the real M5 dataset, each ROW is one item in one store,
    and COLUMNS d_1, d_2, ... d_1941 are the daily sales.
    That's a "wide" format — 1941 columns of sales data.

    For simplicity, we create a "long" format instead
    (one row per item-store-day) which is easier to work with.
    """
    rows = []
    start_date = datetime(2015, 1, 1)

    for item_id in ITEMS:
        dept_id = DEPARTMENTS[item_id]
        cat_id = CATEGORIES[dept_id]

        for store_id in STORES:
            # Generate 30 days of sales
            for day in range(30):
                date = start_date + timedelta(days=day)

                # Base demand depends on department type
                if "HOBBIES" in dept_id:
                    base = np.random.poisson(5)      # Spikey, unpredictable
                elif "HOUSEHOLD" in dept_id:
                    base = np.random.poisson(12)     # Steady, predictable
                elif dept_id == "FOODS_3":
                    base = np.random.poisson(8)      # Moderate, growing
                else:
                    base = np.random.poisson(10)     # Medium

                # Add some seasonality (weekends sell more)
                if date.weekday() >= 5:
                    base = int(base * 1.3)

                # Random event-driven spike
                event = np.random.choice(EVENTS)

                rows.append({
                    "item_id": item_id,
                    "dept_id": dept_id,
                    "cat_id": cat_id,
                    "store_id": store_id,
                    "state_id": store_id.split("_")[0],
                    "date": date.strftime("%Y-%m-%d"),
                    "sales": max(0, base),  # No negative sales
                    "event_name_1": event,
                })

    df = pd.DataFrame(rows)
    df.to_csv("C:/Users/vicky/Desktop/ASU/SooriyaBudhan/Projects/6. agentic-supply-chain-orchestrator/data/raw/sales_train_evaluation.csv", index=False)
    print(f"Created sales data: {len(df)} rows")
    return df


def create_price_data(sales_df):
    """
    Create sell_prices.csv — weekly prices for each item in each store.
    """
    rows = []
    for item_id in ITEMS:
        for store_id in STORES:
            base_price = np.random.uniform(2.0, 50.0)
            for week in range(5):  # 5 weeks covers our 30 days
                # Prices fluctuate slightly week to week
                price = round(base_price * np.random.uniform(0.95, 1.05), 2)
                rows.append({
                    "store_id": store_id,
                    "item_id": item_id,
                    "wm_yr_wk": 11101 + week,  # Walmart week numbering
                    "sell_price": price,
                })

    df = pd.DataFrame(rows)
    df.to_csv("C:/Users/vicky/Desktop/ASU/SooriyaBudhan/Projects/6. agentic-supply-chain-orchestrator/data/raw/sell_prices.csv", index=False)
    print(f"Created price data: {len(df)} rows")
    return df


def create_calendar_data():
    """
    Create calendar.csv — date features including events and SNAP days.

    SNAP = food assistance program days. In the real data, these cause
    demand spikes because people receive benefits and shop more.
    We map SNAP days to "service campaign" demand signals.
    """
    rows = []
    start_date = datetime(2015, 1, 1)

    for day in range(30):
        date = start_date + timedelta(days=day)
        event = np.random.choice(EVENTS)

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + (day // 7),
            "weekday": date.strftime("%A"),
            "wday": date.weekday() + 1,
            "month": date.month,
            "year": date.year,
            "event_name_1": event,
            "event_type_1": "Cultural" if event else None,
            "snap_CA": int(day % 7 < 3),   # SNAP active first 3 days of week
            "snap_TX": int(day % 7 < 2),
            "snap_WI": int(day % 7 < 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv("C:/Users/vicky/Desktop/ASU/SooriyaBudhan/Projects/6. agentic-supply-chain-orchestrator/data/raw/calendar.csv", index=False)
    print(f"Created calendar data: {len(df)} rows")
    return df


if __name__ == "__main__":
    print("=" * 50)
    print("Creating sample M5-style dataset")
    print("=" * 50)
    sales = create_sales_data()
    prices = create_price_data(sales)
    calendar = create_calendar_data()
    print("\nDone! Files saved to data/raw/")
    print(f"  sales:    {len(sales)} rows")
    print(f"  prices:   {len(prices)} rows")
    print(f"  calendar: {len(calendar)} rows")

    # Quick sanity check
    print(f"\nSample sales row:")
    print(sales.iloc[0].to_dict())
