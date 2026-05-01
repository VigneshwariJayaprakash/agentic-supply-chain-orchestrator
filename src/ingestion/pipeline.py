"""
PIPELINE.PY — The Orchestra Conductor
=======================================

WHAT THIS FILE DOES:
    Imagine a factory assembly line:
        Station 1: Raw materials arrive (load CSVs)
        Station 2: Parts are shaped (preprocess / rename columns)
        Station 3: Quality inspector checks each part (schema validation)
        Station 4: Good parts go to the warehouse (save to CSV)
        Station 5: Rejected parts go to the error log (save errors)

    This file IS that assembly line. It calls the preprocessor first,
    then runs every single row through the schema validator.

WHY WE NEED A SEPARATE PIPELINE FILE:
    You might think "why not just put everything in m5_preprocessor.py?"
    Because SEPARATION OF CONCERNS:
        - schema.py knows the RULES (what valid data looks like)
        - m5_preprocessor.py knows the TRANSLATION (M5 → AMAT)
        - pipeline.py knows the PROCESS (what order to do things)

    Each file has ONE job. If the schema rules change, you only edit
    schema.py. If a new data source replaces M5, you only edit the
    preprocessor. The pipeline stays the same.

INTERVIEW TIP:
    "I designed the ETL pipeline with clear separation of concerns:
    schema validation is decoupled from data transformation, which is
    decoupled from pipeline orchestration. This makes each component
    independently testable and the overall system maintainable.
    The pipeline uses an error-accumulation pattern rather than
    fail-fast, so it processes all valid records and reports all
    errors in a single run."
"""

import pandas as pd
import os
import time
from datetime import datetime

# Import our own modules
from src.ingestion.schema import SupplyChainRecord, validate_batch
from src.ingestion.m5_preprocessor import run_preprocessing


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

class PipelineStats:
    """
    Tracks metrics about the pipeline run.

    WHY: In production, you need to know:
        - How many records passed vs failed?
        - How long did it take?
        - What percentage of data is clean?

    These metrics go into your monitoring dashboard and get checked
    in CI/CD (automated testing) before deploying new code.
    """

    def __init__(self):
        self.start_time = time.time()
        self.total_records = 0
        self.valid_records = 0
        self.rejected_records = 0
        self.null_values_found = 0
        self.end_time = None

    @property
    def pass_rate(self) -> float:
        """What percentage of records passed validation?"""
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100

    @property
    def processing_time(self) -> float:
        """How many seconds did the pipeline take?"""
        end = self.end_time or time.time()
        return round(end - self.start_time, 2)

    def finish(self):
        self.end_time = time.time()

    def print_report(self):
        """Print a nice summary of what happened."""
        print(f"\n{'═' * 50}")
        print(f"  PIPELINE RUN REPORT")
        print(f"{'═' * 50}")
        print(f"  Total records:      {self.total_records:>10,}")
        print(f"  ✅ Valid:            {self.valid_records:>10,}")
        print(f"  ❌ Rejected:         {self.rejected_records:>10,}")
        print(f"  Pass rate:          {self.pass_rate:>9.1f}%")
        print(f"  Null values found:  {self.null_values_found:>10,}")
        print(f"  Processing time:    {self.processing_time:>8.2f}s")
        print(f"{'═' * 50}")


# ═══════════════════════════════════════════════════════════════════════════
# THE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    input_dir: str = "data/raw",
    output_dir: str = "data/processed",
    validate: bool = True,
) -> tuple[pd.DataFrame, PipelineStats]:
    """
    Run the complete ETL pipeline from raw M5 data to validated output.

    STEPS:
        1. PREPROCESS: Load raw CSVs, map columns, add quality flags
        2. VALIDATE:   Run every row through schema validation
        3. SEPARATE:   Split into valid records and error records
        4. SAVE:       Write valid records to CSV, errors to error log
        5. REPORT:     Print summary statistics

    PARAMETERS:
        input_dir  : where the raw M5 CSVs live
        output_dir : where to save the processed output
        validate   : if True, run schema validation (set False to skip for speed)

    RETURNS:
        (processed_dataframe, pipeline_statistics)
    """
    stats = PipelineStats()

    # ── STEP 1: PREPROCESS ──────────────────────────────────────────────
    # This calls m5_preprocessor.py which handles:
    #   - Loading the 3 raw CSV files
    #   - Renaming columns (M5 → AMAT schema)
    #   - Mapping departments → demand types
    #   - Mapping stores → regions
    #   - Adding data quality flags
    print("\n📥 STEP 1: Preprocessing raw data...")
    preprocessed_df = run_preprocessing(
        input_dir=input_dir,
        output_path=os.path.join(output_dir, "m5_amat_mapped.csv"),
    )
    stats.total_records = len(preprocessed_df)
    stats.null_values_found = int(preprocessed_df.isnull().sum().sum())

    if not validate:
        stats.valid_records = stats.total_records
        stats.finish()
        stats.print_report()
        return preprocessed_df, stats

    # ── STEP 2: VALIDATE ────────────────────────────────────────────────
    # Now we check EVERY row against our schema rules.
    # This is like a quality inspector examining each part on the line.
    print("\n🔍 STEP 2: Validating against schema...")

    # Convert DataFrame rows to dictionaries for validation
    # We need to handle NaN values (which pandas uses for missing data)
    records_to_validate = []
    for _, row in preprocessed_df.iterrows():
        record = {
            "part_number": str(row.get("part_number", "")),
            "demand_type": str(row.get("demand_type", "")),
            "region": str(row.get("region", "")),
            "demand_quantity": int(row.get("demand_quantity", 0)),
            "transaction_date": str(row.get("transaction_date", "")),
            "unit_cost": float(row.get("unit_cost", 0.0)),
            "part_description": str(row.get("part_description", "")),
            "npi_signal": str(row["npi_signal"]) if pd.notna(row.get("npi_signal")) else None,
        }
        records_to_validate.append(record)

    valid_records, errors = validate_batch(records_to_validate)

    stats.valid_records = len(valid_records)
    stats.rejected_records = len(errors)

    # ── STEP 3: SAVE VALID RECORDS ──────────────────────────────────────
    if valid_records:
        valid_df = pd.DataFrame([r.to_dict() for r in valid_records])
        valid_path = os.path.join(output_dir, "m5_amat_validated.csv")
        valid_df.to_csv(valid_path, index=False)
        print(f"  ✅ Saved {len(valid_records):,} valid records to {valid_path}")
    else:
        valid_df = pd.DataFrame()
        print("  ⚠️  No valid records!")

    # ── STEP 4: SAVE ERROR LOG ──────────────────────────────────────────
    # In production, you'd send these to a monitoring system (like Datadog)
    # so the team gets alerted when data quality drops.
    if errors:
        error_df = pd.DataFrame(errors)
        error_path = os.path.join(output_dir, "validation_errors.csv")
        error_df.to_csv(error_path, index=False)
        print(f"  ❌ Saved {len(errors):,} errors to {error_path}")

        # Show first 3 errors so you can see what went wrong
        print(f"\n  First 3 errors:")
        for err in errors[:3]:
            print(f"    Row {err['row']}: {err['error'][:100]}")

    # ── STEP 5: REPORT ──────────────────────────────────────────────────
    stats.finish()
    stats.print_report()

    return valid_df, stats


# ─── RUN IT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, stats = run_pipeline()
    print(f"\nPipeline complete! {stats.valid_records:,} records ready for ML models.")
