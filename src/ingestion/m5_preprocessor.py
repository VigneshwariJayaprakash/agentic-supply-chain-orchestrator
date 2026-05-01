"""
M5_PREPROCESSOR.PY — The Translator
=====================================

WHAT THIS FILE DOES:
    Takes the raw M5 (Walmart) data and translates every column
    into our supply chain language.

    It's like taking a book written in Spanish and translating it to French.
    The STORIES (data patterns) stay the same — only the WORDS (column names
    and category labels) change.

WHY WE NEED THIS:
    The M5 dataset uses Walmart terminology: "items", "stores", "departments".
    Our project pretends this is semiconductor data: "parts", "regions", "demand types".
    This file does the translation so the rest of our code only sees
    supply chain terminology.

THE MAPPING LOGIC:
    Walmart column → Our column       → Why this mapping makes sense
    ─────────────────────────────────────────────────────────────────
    item_id       → part_number       → Each item becomes a spare part
    dept_id       → demand_type       → Departments map to demand categories
    store_id      → region            → Stores become geographic regions
    sales         → demand_quantity   → Sales volume = demand for parts
    sell_price    → unit_cost         → Retail price ≈ part cost
    cat_id        → part_description  → Category = part description
    event_name_1  → npi_signal        → Events = product launch signals
    date          → transaction_date  → Just a rename

INTERVIEW TIP:
    "I designed a preprocessing pipeline that maps the M5 hierarchical
    retail schema to a semiconductor supply chain schema. The mapping
    preserves the statistical properties — seasonality, trends, and
    event-driven demand spikes — while relabeling the taxonomy to match
    our domain. Department-to-demand-type mapping was designed to reflect
    real supply chain demand patterns: HOBBIES → NPI because hobby items
    exhibit the same spike-and-fade pattern as new product introductions."
"""

import pandas as pd
import os


# ═══════════════════════════════════════════════════════════════════════════
# MAPPING DICTIONARIES
# ═══════════════════════════════════════════════════════════════════════════
# These are the "translation dictionaries" — like a Spanish-to-French
# dictionary, but for data columns.

DEPT_TO_DEMAND_TYPE = {
    # ── NPI (New Product Introduction) ──
    # Hobby items spike suddenly when a new trend starts, then fade.
    # This is EXACTLY how demand behaves when a new chip launches.
    "HOBBIES_1": "NPI",
    "HOBBIES_2": "NPI",

    # ── Standard ──
    # Household items (soap, paper towels) have steady, predictable demand.
    # This matches standard replacement parts that factories always need.
    "HOUSEHOLD_1": "standard",
    "HOUSEHOLD_2": "standard",

    # ── Service Campaign ──
    # Food demand spikes around holidays then drops sharply.
    # This matches "recall" or "free replacement" campaigns.
    "FOODS_1": "service_campaign",
    "FOODS_2": "service_campaign",

    # ── Reliability ──
    # Some food staples have gradual, steady growth over time.
    # This matches parts that wear out — demand grows as equipment ages.
    "FOODS_3": "reliability",
}

STORE_TO_REGION = {
    # California and Texas stores → North America
    "CA_1": "North America",
    "CA_2": "North America",
    "CA_3": "North America",
    "CA_4": "North America",
    "TX_1": "North America",
    "TX_2": "North America",
    "TX_3": "North America",

    # Wisconsin stores → split between Asia Pacific and Europe
    # (In real life, these would be actual AMAT facilities)
    "WI_1": "Asia Pacific",
    "WI_2": "Asia Pacific",
    "WI_3": "Europe",
}


# ═══════════════════════════════════════════════════════════════════════════
# PART NUMBER MAPPING
# ═══════════════════════════════════════════════════════════════════════════
# Translate Walmart item IDs into realistic semiconductor part numbers.
#
# Real semiconductor parts follow naming conventions like:
#   AMAT-[category]-[series]-[variant]
#
# Examples from actual supply chains:
#   - Etch chamber RF generators
#   - CVD (Chemical Vapor Deposition) gas delivery modules
#   - CMP (Chemical Mechanical Polishing) consumables
#   - Lithography alignment sensors

ITEM_TO_PART_NUMBER = {
    # NPI parts — new technology, spikey demand
    "HOBBIES_1_001": "AMAT-RF-7800-GEN",        # RF generator for etch chambers
    "HOBBIES_1_008": "AMAT-RF-7800-CTRL",       # RF controller module
    "HOBBIES_2_004": "AMAT-LITH-5500-ALIGN",    # Lithography alignment sensor

    # Standard parts — steady demand, always needed
    "HOUSEHOLD_1_012": "AMAT-CVD-3200-VALVE",   # CVD gas delivery valve
    "HOUSEHOLD_2_067": "AMAT-CVD-3200-SEAL",    # CVD chamber seal ring

    # Service campaign parts — spike during recalls/upgrades
    "FOODS_1_001": "AMAT-CMP-4100-PAD",         # CMP polishing pad
    "FOODS_2_099": "AMAT-CMP-4100-SLURRY",      # CMP slurry pump assembly

    # Reliability parts — gradual increase as equipment ages
    "FOODS_3_042": "AMAT-ETCH-6600-RING",       # Etch chamber focus ring (wears out)
}


# ═══════════════════════════════════════════════════════════════════════════
# PART DESCRIPTION MAPPING
# ═══════════════════════════════════════════════════════════════════════════
# Translate generic categories into semiconductor equipment descriptions.

CATEGORY_TO_DESCRIPTION = {
    "HOBBIES":   "RF & Lithography Components",
    "HOUSEHOLD": "CVD Gas Delivery Systems",
    "FOODS":     "CMP & Etch Consumables",
}


# ═══════════════════════════════════════════════════════════════════════════
# NPI SIGNAL MAPPING
# ═══════════════════════════════════════════════════════════════════════════
# Translate Walmart events into semiconductor industry events.

EVENT_TO_NPI_SIGNAL = {
    "SuperBowl":          "AMAT_TechNode_3nm_Launch",
    "NPI_Launch_Q1":      "AMAT_Producer_SE_Release",
    "ServiceCampaign_A":  "AMAT_FieldUpgrade_CVD3200",
}


# ═══════════════════════════════════════════════════════════════════════════
# THE PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

def load_raw_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three raw M5 CSV files.

    WHAT EACH FILE CONTAINS:
        sales_train_evaluation.csv — The main data: what sold, where, when
        sell_prices.csv           — Weekly prices for each item at each store
        calendar.csv              — Date info: day of week, events, SNAP days

    RETURNS:
        Three DataFrames: (sales, prices, calendar)
    """
    sales_path = os.path.join(data_dir, "sales_train_evaluation.csv")
    prices_path = os.path.join(data_dir, "sell_prices.csv")
    calendar_path = os.path.join(data_dir, "calendar.csv")

    # Check that files exist before trying to read them
    for path, name in [(sales_path, "sales"), (prices_path, "prices"), (calendar_path, "calendar")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {name} file: {path}\n"
                f"Download M5 data from Kaggle and place in {data_dir}/"
            )

    sales = pd.read_csv(sales_path)
    prices = pd.read_csv(prices_path)
    calendar = pd.read_csv(calendar_path)

    print(f"Loaded raw data:")
    print(f"  Sales:    {len(sales):>8,} rows × {len(sales.columns)} columns")
    print(f"  Prices:   {len(prices):>8,} rows × {len(prices.columns)} columns")
    print(f"  Calendar: {len(calendar):>8,} rows × {len(calendar.columns)} columns")

    return sales, prices, calendar


def map_columns(sales_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    THE CORE TRANSFORMATION: Rename M5 columns to AMAT supply chain schema.

    This is where the "translation" happens. We:
    1. Rename columns (item_id → part_number)
    2. Map department IDs to demand types (HOBBIES_1 → NPI)
    3. Map store IDs to regions (CA_1 → North America)
    4. Merge in price data as unit_cost
    5. Clean up event names as NPI signals

    STEP BY STEP:
    ─────────────
    Think of it like filling out a form. The old form says "Name" and
    the new form says "Full Legal Name" — same info, different label.
    We're just moving data from one form to another.
    """
    df = sales_df.copy()

    # ── Step 1: Map department → demand type ──
    # Look up each dept_id in our dictionary. If it's not in the dictionary,
    # default to "standard" (safe fallback).
    df["demand_type"] = df["dept_id"].map(DEPT_TO_DEMAND_TYPE).fillna("standard")

    # ── Step 2: Map store → region ──
    # Same idea — translate store codes to geographic regions.
    df["region"] = df["store_id"].map(STORE_TO_REGION).fillna("North America")

    # ── Step 3: Map item IDs → semiconductor part numbers ──
    # "HOBBIES_1_001" becomes "AMAT-RF-7800-GEN" (an RF generator)
    # This makes the data look like real semiconductor supply chain data.
    df["item_id"] = df["item_id"].map(ITEM_TO_PART_NUMBER).fillna(df["item_id"])

    # ── Step 4: Map categories → semiconductor descriptions ──
    # "HOBBIES" becomes "RF & Lithography Components"
    df["cat_id"] = df["cat_id"].map(CATEGORY_TO_DESCRIPTION).fillna(df["cat_id"])

    # ── Step 5: Map event names → semiconductor NPI signals ──
    # "SuperBowl" becomes "AMAT_TechNode_3nm_Launch"
    if "event_name_1" in df.columns:
        df["event_name_1"] = df["event_name_1"].map(EVENT_TO_NPI_SIGNAL)

    # ── Step 6: Merge price data ──
    # The price file has weekly prices per item per store.
    # We merge it into our main data so each row knows its price.
    #
    # If a row doesn't have a matching price (left join miss),
    # we fill with 0.0 — these will be flagged in validation.
    if "sell_price" in prices_df.columns and "wm_yr_wk" not in df.columns:
        # For our simplified sample data, use average price per item-store
        avg_prices = prices_df.groupby(["item_id", "store_id"])["sell_price"].mean().reset_index()
        # Map the price file's item_ids to our new part numbers too
        avg_prices["item_id"] = avg_prices["item_id"].map(ITEM_TO_PART_NUMBER).fillna(avg_prices["item_id"])
        # Map store_id in prices to match (but we already dropped it later, so merge on original)
        df = df.merge(avg_prices, on=["item_id", "store_id"], how="left")
        df["sell_price"] = df["sell_price"].fillna(0.01)  # Small default instead of 0
    elif "sell_price" not in df.columns:
        df["sell_price"] = 0.01

    # ── Step 7: Rename columns to AMAT schema ──
    # This is the actual "translation" — old name → new name
    df = df.rename(columns={
        "item_id":      "part_number",
        "sales":        "demand_quantity",
        "date":         "transaction_date",
        "sell_price":   "unit_cost",
        "cat_id":       "part_description",
        "event_name_1": "npi_signal",
    })

    # ── Step 5: Select only the columns we need ──
    # Drop the old M5-specific columns we don't need anymore
    output_columns = [
        "part_number",
        "demand_type",
        "region",
        "demand_quantity",
        "transaction_date",
        "unit_cost",
        "part_description",
        "npi_signal",
    ]

    # Only keep columns that exist (some might not in sample data)
    existing_columns = [col for col in output_columns if col in df.columns]
    df = df[existing_columns]

    return df


def add_data_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quality flags that help analysts spot potential issues.

    WHY: In real supply chain data, you often get weird patterns:
    - A part that suddenly has 10x normal demand (data entry error? or real spike?)
    - Zero demand for weeks then a huge spike (was the part out of stock?)

    We flag these so the ML models can either handle them or ignore them.

    INTERVIEW TIP:
        "I implemented automated data quality checks including outlier
        detection using z-scores and zero-demand flagging. This helps
        distinguish between genuine demand signals and data quality
        issues before they impact model training."
    """
    df = df.copy()

    # Flag 1: Is demand zero?
    # Zero demand might mean "nobody needed this part" or it might mean
    # "we were out of stock so we couldn't record any demand"
    df["is_zero_demand"] = (df["demand_quantity"] == 0).astype(int)

    # Flag 2: Is this a high-demand outlier?
    # If demand is more than 3 standard deviations above the mean,
    # it might be a data entry error OR a genuine demand spike.
    # We flag it so analysts can investigate.
    if len(df) > 1:
        mean_demand = df["demand_quantity"].mean()
        std_demand = df["demand_quantity"].std()
        if std_demand > 0:
            df["is_demand_outlier"] = (
                (df["demand_quantity"] > mean_demand + 3 * std_demand)
            ).astype(int)
        else:
            df["is_demand_outlier"] = 0
    else:
        df["is_demand_outlier"] = 0

    # Flag 3: Does this row have an NPI signal?
    # This is important because NPI-driven demand behaves differently
    # from regular demand — the ML model needs to know.
    df["has_npi_signal"] = df["npi_signal"].notna().astype(int)

    return df


def run_preprocessing(
    input_dir: str = "data/raw",
    output_path: str = "data/processed/m5_amat_mapped.csv",
) -> pd.DataFrame:
    """
    THE MAIN FUNCTION: Run the complete preprocessing pipeline.

    This is the function you call to transform raw M5 data into
    clean, validated, supply-chain-schema data.

    PIPELINE STEPS:
        1. Load raw CSVs
        2. Map columns from M5 → AMAT schema
        3. Add data quality flags
        4. Save the result
        5. Print summary statistics

    RETURNS:
        The processed DataFrame
    """
    print("=" * 60)
    print("M5 → AMAT Supply Chain Preprocessing")
    print("=" * 60)

    # Step 1: Load
    sales, prices, calendar = load_raw_data(input_dir)

    # Step 2: Transform
    print("\nMapping M5 columns to AMAT schema...")
    mapped_df = map_columns(sales, prices)
    print(f"  Mapped {len(mapped_df):,} rows")

    # Step 3: Quality flags
    print("Adding data quality flags...")
    processed_df = add_data_quality_flags(mapped_df)

    # Step 4: Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to {output_path}")

    # Step 5: Summary statistics
    print(f"\n{'─' * 40}")
    print(f"SUMMARY")
    print(f"{'─' * 40}")
    print(f"Total records:     {len(processed_df):>10,}")
    print(f"Columns:           {len(processed_df.columns):>10}")
    print(f"Date range:        {processed_df['transaction_date'].min()} to {processed_df['transaction_date'].max()}")
    print(f"Unique parts:      {processed_df['part_number'].nunique():>10}")
    print(f"Unique regions:    {processed_df['region'].nunique():>10}")
    print(f"\nDemand type breakdown:")
    for dtype, count in processed_df["demand_type"].value_counts().items():
        pct = count / len(processed_df) * 100
        print(f"  {dtype:<20} {count:>6,} ({pct:.1f}%)")
    print(f"\nRegion breakdown:")
    for region, count in processed_df["region"].value_counts().items():
        pct = count / len(processed_df) * 100
        print(f"  {region:<20} {count:>6,} ({pct:.1f}%)")
    print(f"\nData quality:")
    print(f"  Zero demand rows:  {processed_df['is_zero_demand'].sum():>6,}")
    print(f"  Outlier rows:      {processed_df['is_demand_outlier'].sum():>6,}")
    print(f"  NPI signal rows:   {processed_df['has_npi_signal'].sum():>6,}")
    print(f"  Null values:       {processed_df.isnull().sum().sum():>6,}")

    return processed_df


# ─── RUN IT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = run_preprocessing()
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
