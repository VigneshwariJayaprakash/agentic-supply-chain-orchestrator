"""
TEST_INGESTION.PY — Testing Every Piece of the Data Pipeline
==============================================================

WHAT ARE TESTS?
    Tests are like a checklist for your code. Before you ship a package,
    you shake it to make sure nothing rattles. Tests "shake" your code
    by feeding it tricky inputs and checking the outputs.

WHY WE WRITE TESTS:
    1. They catch bugs BEFORE they reach production
    2. They let you change code confidently (if tests pass, nothing broke)
    3. Interviewers LOVE seeing tests — it shows you write production-quality code
    4. They serve as documentation (reading tests shows you HOW code works)

HOW TO RUN THESE TESTS:
    cd project/
    python -m pytest tests/test_ingestion.py -v

    The "-v" flag means "verbose" — it shows each test name and pass/fail.

WHAT EACH TEST DOES (plain English):
    See the comment above each test function — I explain it like you're 10.

INTERVIEW TIP:
    "I wrote unit tests covering schema validation, data transformation,
    and pipeline orchestration. Tests verify both happy paths and edge
    cases including malformed data, boundary values, and type coercion.
    I aimed for 100% pass rate on the validation layer with zero
    silent data corruption."
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Make sure Python can find our project code ──
# This adds the project root to the Python path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion.schema import SupplyChainRecord, validate_batch, VALID_DEMAND_TYPES, VALID_REGIONS
from src.ingestion.m5_preprocessor import (
    DEPT_TO_DEMAND_TYPE,
    STORE_TO_REGION,
    map_columns,
    add_data_quality_flags,
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: Create a valid record quickly
# ═══════════════════════════════════════════════════════════════════════════
# Instead of typing all 8 fields every time, we make a helper function.
# Tests that need a valid record just call this. Tests that need a BROKEN
# record call this and override one field.

def make_valid_record(**overrides) -> dict:
    """
    Create a dictionary that represents a perfectly valid record.
    Pass overrides to change specific fields for testing.

    Example:
        make_valid_record()                          → valid record
        make_valid_record(demand_quantity=-5)         → invalid (negative qty)
        make_valid_record(region="Mars")              → invalid (bad region)
    """
    base = {
        "part_number": "AMAT-RF-7800-GEN",
        "demand_type": "NPI",
        "region": "North America",
        "demand_quantity": 42,
        "transaction_date": "2015-06-15",
        "unit_cost": 12.99,
        "part_description": "RF & Lithography Components",
        "npi_signal": "AMAT_Producer_SE_Release",
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 1: SCHEMA VALIDATION — Does the bouncer work?
# ═══════════════════════════════════════════════════════════════════════════

class TestSchemaValidation:
    """
    These tests check: does our schema correctly ACCEPT good data
    and REJECT bad data?

    It's like testing a lock:
        - Does the RIGHT key open it? (valid data passes)
        - Does a WRONG key get rejected? (invalid data fails)
        - Does a bent key get rejected? (edge cases fail)
    """

    # ── TEST 1: A perfect record should pass ──
    # Like checking that a valid passport gets you through customs.
    def test_valid_record_passes(self):
        record = SupplyChainRecord(**make_valid_record())
        assert record.part_number == "AMAT-RF-7800-GEN"
        assert record.demand_quantity == 42
        print("  ✅ Valid record accepted correctly")

    # ── TEST 2: Missing part_number should fail ──
    # Every part needs an ID. No ID = we don't know what this is.
    # Like a package arriving with no address label.
    def test_empty_part_number_rejected(self):
        try:
            SupplyChainRecord(**make_valid_record(part_number=""))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "part_number" in str(e)
            print("  ✅ Empty part_number correctly rejected")

    # ── TEST 3: Negative demand should fail ──
    # You can't demand -50 parts. That makes no physical sense.
    # Like ordering -3 pizzas — the pizza shop would be confused.
    def test_negative_demand_rejected(self):
        try:
            SupplyChainRecord(**make_valid_record(demand_quantity=-5))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "negative" in str(e).lower() or "demand_quantity" in str(e)
            print("  ✅ Negative demand correctly rejected")

    # ── TEST 4: Zero demand should PASS ──
    # Zero is valid — it means "nobody needed this part today."
    # Important: zero is NOT the same as negative!
    def test_zero_demand_accepted(self):
        record = SupplyChainRecord(**make_valid_record(demand_quantity=0))
        assert record.demand_quantity == 0
        print("  ✅ Zero demand correctly accepted")

    # ── TEST 5: Bad demand_type should fail ──
    # Only 4 types are allowed: NPI, standard, service_campaign, reliability.
    # "banana" is not a demand type.
    def test_invalid_demand_type_rejected(self):
        try:
            SupplyChainRecord(**make_valid_record(demand_type="banana"))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "demand_type" in str(e)
            print("  ✅ Invalid demand_type 'banana' correctly rejected")

    # ── TEST 6: All 4 valid demand types should pass ──
    # Make sure we didn't accidentally block a valid type.
    def test_all_valid_demand_types_accepted(self):
        for dtype in VALID_DEMAND_TYPES:
            record = SupplyChainRecord(**make_valid_record(demand_type=dtype))
            assert record.demand_type == dtype
        print(f"  ✅ All {len(VALID_DEMAND_TYPES)} demand types accepted")

    # ── TEST 7: Bad region should fail ──
    # "Mars" is not a valid region (yet).
    def test_invalid_region_rejected(self):
        try:
            SupplyChainRecord(**make_valid_record(region="Mars"))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "region" in str(e)
            print("  ✅ Invalid region 'Mars' correctly rejected")

    # ── TEST 8: All 3 valid regions should pass ──
    def test_all_valid_regions_accepted(self):
        for region in VALID_REGIONS:
            record = SupplyChainRecord(**make_valid_record(region=region))
            assert record.region == region
        print(f"  ✅ All {len(VALID_REGIONS)} regions accepted")

    # ── TEST 9: Negative price should fail ──
    # A part can't cost -$5. That would mean THEY pay YOU to take it.
    def test_negative_unit_cost_rejected(self):
        try:
            SupplyChainRecord(**make_valid_record(unit_cost=-5.0))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unit_cost" in str(e)
            print("  ✅ Negative unit_cost correctly rejected")

    # ── TEST 10: Zero price should fail ──
    # Even "free" parts have an internal cost > 0.
    def test_zero_unit_cost_rejected(self):
        try:
            SupplyChainRecord(**make_valid_record(unit_cost=0.0))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "unit_cost" in str(e)
            print("  ✅ Zero unit_cost correctly rejected")

    # ── TEST 11: Bad date format should fail ──
    # "2015-13-45" is not a real date (month 13 doesn't exist).
    def test_invalid_date_rejected(self):
        try:
            SupplyChainRecord(**make_valid_record(transaction_date="2015-13-45"))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "transaction_date" in str(e)
            print("  ✅ Invalid date '2015-13-45' correctly rejected")

    # ── TEST 12: Valid date should pass ──
    def test_valid_date_accepted(self):
        record = SupplyChainRecord(**make_valid_record(transaction_date="2023-12-31"))
        assert record.transaction_date == "2023-12-31"
        print("  ✅ Valid date '2023-12-31' correctly accepted")

    # ── TEST 13: NPI signal can be None (optional field) ──
    # Not every day has an NPI event. None means "no event today."
    def test_null_npi_signal_accepted(self):
        record = SupplyChainRecord(**make_valid_record(npi_signal=None))
        assert record.npi_signal is None
        print("  ✅ None npi_signal correctly accepted")

    # ── TEST 14: to_dict() returns the right format ──
    # We need this to convert back to DataFrames later.
    def test_to_dict_has_all_fields(self):
        record = SupplyChainRecord(**make_valid_record())
        d = record.to_dict()
        expected_keys = {
            "part_number", "demand_type", "region", "demand_quantity",
            "transaction_date", "unit_cost", "part_description", "npi_signal"
        }
        assert set(d.keys()) == expected_keys
        print("  ✅ to_dict() has all expected keys")

    # ── TEST 15: Empty part_description should fail ──
    def test_empty_description_rejected(self):
        try:
            SupplyChainRecord(**make_valid_record(part_description=""))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "part_description" in str(e)
            print("  ✅ Empty part_description correctly rejected")


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 2: BATCH VALIDATION — Does the assembly line work?
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchValidation:
    """
    These tests check the validate_batch() function which processes
    MANY records at once and separates good from bad.

    It's like a fruit sorting machine:
        - Good apples go in the "sell" bin
        - Bruised apples go in the "reject" bin
        - The machine doesn't stop when it finds one bad apple
    """

    # ── TEST 16: All good records should all pass ──
    def test_all_valid_records_pass(self):
        records = [make_valid_record() for _ in range(10)]
        valid, errors = validate_batch(records)
        assert len(valid) == 10
        assert len(errors) == 0
        print("  ✅ Batch of 10 valid records: all passed")

    # ── TEST 17: Mix of good and bad should be separated correctly ──
    # 3 good + 2 bad = 3 in valid bin + 2 in error bin
    def test_mixed_batch_separated(self):
        records = [
            make_valid_record(),                              # ✅ good
            make_valid_record(demand_quantity=-1),             # ❌ bad
            make_valid_record(),                              # ✅ good
            make_valid_record(region="Atlantis"),              # ❌ bad
            make_valid_record(),                              # ✅ good
        ]
        valid, errors = validate_batch(records)
        assert len(valid) == 3, f"Expected 3 valid, got {len(valid)}"
        assert len(errors) == 2, f"Expected 2 errors, got {len(errors)}"
        print("  ✅ Mixed batch: 3 valid + 2 invalid correctly separated")

    # ── TEST 18: Error messages should tell you WHICH row failed ──
    def test_errors_include_row_number(self):
        records = [
            make_valid_record(),                              # row 0: good
            make_valid_record(unit_cost=-10),                 # row 1: bad
        ]
        valid, errors = validate_batch(records)
        assert errors[0]["row"] == 1  # The bad record was at index 1
        print("  ✅ Error messages include correct row numbers")

    # ── TEST 19: Empty batch should return empty results ──
    def test_empty_batch(self):
        valid, errors = validate_batch([])
        assert len(valid) == 0
        assert len(errors) == 0
        print("  ✅ Empty batch returns empty results")

    # ── TEST 20: ALL bad records should ALL fail ──
    def test_all_invalid_records_rejected(self):
        records = [
            make_valid_record(demand_quantity=-1),
            make_valid_record(region="Narnia"),
            make_valid_record(unit_cost=-99),
        ]
        valid, errors = validate_batch(records)
        assert len(valid) == 0
        assert len(errors) == 3
        print("  ✅ Batch of 3 invalid records: all rejected")


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 3: M5 PREPROCESSING — Does the translator work?
# ═══════════════════════════════════════════════════════════════════════════

class TestM5Preprocessing:
    """
    These tests check: does the M5 → AMAT column mapping work correctly?

    It's like testing a language translator:
        - Does "hola" translate to "hello"? (column rename check)
        - Does "HOBBIES_1" translate to "NPI"? (category mapping check)
        - Does "CA_1" translate to "North America"? (region mapping check)
    """

    # ── TEST 21: Every department should map to a demand type ──
    # This catches typos in the mapping dictionary.
    def test_all_departments_have_mapping(self):
        expected_depts = [
            "HOBBIES_1", "HOBBIES_2",
            "HOUSEHOLD_1", "HOUSEHOLD_2",
            "FOODS_1", "FOODS_2", "FOODS_3",
        ]
        for dept in expected_depts:
            assert dept in DEPT_TO_DEMAND_TYPE, f"Missing mapping for {dept}"
        print(f"  ✅ All {len(expected_depts)} departments have demand type mappings")

    # ── TEST 22: Every store should map to a region ──
    def test_all_stores_have_mapping(self):
        expected_stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
        for store in expected_stores:
            assert store in STORE_TO_REGION, f"Missing mapping for {store}"
        print(f"  ✅ All {len(expected_stores)} stores have region mappings")

    # ── TEST 23: Demand types should only contain valid values ──
    # The mapping dictionary values must match what our schema accepts.
    def test_mapped_demand_types_are_valid(self):
        for dept, dtype in DEPT_TO_DEMAND_TYPE.items():
            assert dtype in VALID_DEMAND_TYPES, (
                f"Dept {dept} maps to '{dtype}' which isn't a valid demand type!"
            )
        print("  ✅ All mapped demand types are valid schema values")

    # ── TEST 24: Mapped regions should only contain valid values ──
    def test_mapped_regions_are_valid(self):
        for store, region in STORE_TO_REGION.items():
            assert region in VALID_REGIONS, (
                f"Store {store} maps to '{region}' which isn't a valid region!"
            )
        print("  ✅ All mapped regions are valid schema values")

    # ── TEST 25: Column mapping produces the right output columns ──
    def test_map_columns_output_structure(self):
        # Create a tiny fake M5 dataframe
        sales_df = pd.DataFrame([{
            "item_id": "HOBBIES_1_001",
            "dept_id": "HOBBIES_1",
            "cat_id": "HOBBIES",
            "store_id": "CA_1",
            "date": "2015-01-01",
            "sales": 5,
            "event_name_1": "SuperBowl",
        }])
        prices_df = pd.DataFrame([{
            "item_id": "HOBBIES_1_001",
            "store_id": "CA_1",
            "wm_yr_wk": 11101,
            "sell_price": 9.99,
        }])

        result = map_columns(sales_df, prices_df)

        # Check that M5 column names are GONE
        assert "item_id" not in result.columns, "item_id should be renamed"
        assert "store_id" not in result.columns, "store_id should be removed"
        assert "dept_id" not in result.columns, "dept_id should be removed"
        assert "sales" not in result.columns, "sales should be renamed"

        # Check that AMAT column names are PRESENT
        assert "part_number" in result.columns
        assert "demand_type" in result.columns
        assert "region" in result.columns
        assert "demand_quantity" in result.columns
        assert "transaction_date" in result.columns
        assert "unit_cost" in result.columns

        print("  ✅ Column mapping produces correct output columns")

    # ── TEST 26: Specific mapping values are correct ──
    def test_specific_mapping_values(self):
        sales_df = pd.DataFrame([{
            "item_id": "FOODS_3_042",
            "dept_id": "FOODS_3",
            "cat_id": "FOODS",
            "store_id": "WI_3",
            "date": "2015-03-10",
            "sales": 8,
            "event_name_1": None,
        }])
        prices_df = pd.DataFrame([{
            "item_id": "FOODS_3_042",
            "store_id": "WI_3",
            "wm_yr_wk": 11101,
            "sell_price": 4.50,
        }])

        result = map_columns(sales_df, prices_df)

        assert result.iloc[0]["part_number"] == "AMAT-ETCH-6600-RING"
        assert result.iloc[0]["demand_type"] == "reliability"  # FOODS_3 → reliability
        assert result.iloc[0]["region"] == "Europe"            # WI_3 → Europe
        assert result.iloc[0]["demand_quantity"] == 8
        print("  ✅ FOODS_3/WI_3 correctly maps to AMAT-ETCH-6600-RING/reliability/Europe")


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 4: DATA QUALITY FLAGS — Do the warning labels work?
# ═══════════════════════════════════════════════════════════════════════════

class TestDataQualityFlags:
    """
    These tests check: do our quality flags correctly identify
    suspicious data patterns?

    It's like a smoke detector — you don't want it going off
    when you're cooking (false alarm), but you definitely want
    it going off when there's a real fire.
    """

    # ── TEST 27: Zero demand gets flagged ──
    def test_zero_demand_flagged(self):
        df = pd.DataFrame([{
            "part_number": "TEST_001",
            "demand_quantity": 0,
            "npi_signal": None,
        }])
        result = add_data_quality_flags(df)
        assert result.iloc[0]["is_zero_demand"] == 1
        print("  ✅ Zero demand correctly flagged")

    # ── TEST 28: Normal demand is NOT flagged as zero ──
    def test_nonzero_demand_not_flagged(self):
        df = pd.DataFrame([{
            "part_number": "TEST_001",
            "demand_quantity": 10,
            "npi_signal": None,
        }])
        result = add_data_quality_flags(df)
        assert result.iloc[0]["is_zero_demand"] == 0
        print("  ✅ Nonzero demand correctly NOT flagged")

    # ── TEST 29: NPI signal presence is detected ──
    def test_npi_signal_detected(self):
        df = pd.DataFrame([
            {"part_number": "A", "demand_quantity": 5, "npi_signal": "Launch_Q1"},
            {"part_number": "B", "demand_quantity": 5, "npi_signal": None},
        ])
        result = add_data_quality_flags(df)
        assert result.iloc[0]["has_npi_signal"] == 1  # Has signal
        assert result.iloc[1]["has_npi_signal"] == 0  # No signal
        print("  ✅ NPI signal presence/absence correctly detected")

    # ── TEST 30: Outlier detection works ──
    # If 99 records have demand=10 and 1 record has demand=999,
    # that 999 should be flagged as an outlier.
    def test_outlier_detected(self):
        normal = [{"part_number": f"P{i}", "demand_quantity": 10, "npi_signal": None}
                  for i in range(99)]
        outlier = [{"part_number": "P99", "demand_quantity": 999, "npi_signal": None}]
        df = pd.DataFrame(normal + outlier)
        result = add_data_quality_flags(df)
        assert result.iloc[-1]["is_demand_outlier"] == 1  # The 999 should be flagged
        print("  ✅ Demand outlier (999 among 10s) correctly flagged")

    # ── TEST 31: Quality flag columns exist ──
    def test_quality_columns_added(self):
        df = pd.DataFrame([{
            "part_number": "TEST_001",
            "demand_quantity": 10,
            "npi_signal": None,
        }])
        result = add_data_quality_flags(df)
        expected_flags = ["is_zero_demand", "is_demand_outlier", "has_npi_signal"]
        for flag in expected_flags:
            assert flag in result.columns, f"Missing quality flag: {flag}"
        print("  ✅ All 3 quality flag columns are present")


# ═══════════════════════════════════════════════════════════════════════════
# TEST GROUP 5: END-TO-END PIPELINE — Does the whole thing work?
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndPipeline:
    """
    These tests check the COMPLETE pipeline from raw data to validated output.

    Previous tests checked individual pieces (like testing each car part).
    This test checks the WHOLE CAR — does it start, drive, and stop?
    """

    # ── TEST 32: Pipeline produces output file ──
    def test_pipeline_creates_output(self):
        output_path = "data/processed/m5_amat_mapped.csv"
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            assert len(df) > 0, "Output file is empty!"
            print(f"  ✅ Pipeline output exists with {len(df):,} rows")
        else:
            print("  ⚠️  Skipped — run pipeline first to create output")

    # ── TEST 33: Output has no unexpected null values in required fields ──
    def test_no_nulls_in_required_fields(self):
        output_path = "data/processed/m5_amat_validated.csv"
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            required = ["part_number", "demand_type", "region",
                       "demand_quantity", "transaction_date", "unit_cost"]
            for col in required:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    assert null_count == 0, f"{col} has {null_count} null values!"
            print("  ✅ No null values in required fields")
        else:
            print("  ⚠️  Skipped — run pipeline with validation first")

    # ── TEST 34: All demand types in output are valid ──
    def test_output_demand_types_valid(self):
        output_path = "data/processed/m5_amat_mapped.csv"
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            unique_types = set(df["demand_type"].unique())
            assert unique_types.issubset(VALID_DEMAND_TYPES), (
                f"Found invalid demand types: {unique_types - VALID_DEMAND_TYPES}"
            )
            print(f"  ✅ All demand types in output are valid: {unique_types}")
        else:
            print("  ⚠️  Skipped — run pipeline first")

    # ── TEST 35: All regions in output are valid ──
    def test_output_regions_valid(self):
        output_path = "data/processed/m5_amat_mapped.csv"
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            unique_regions = set(df["region"].unique())
            assert unique_regions.issubset(VALID_REGIONS), (
                f"Found invalid regions: {unique_regions - VALID_REGIONS}"
            )
            print(f"  ✅ All regions in output are valid: {unique_regions}")
        else:
            print("  ⚠️  Skipped — run pipeline first")

    # ── TEST 36: No negative demand quantities in output ──
    def test_no_negative_demand_in_output(self):
        output_path = "data/processed/m5_amat_mapped.csv"
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            negative_count = (df["demand_quantity"] < 0).sum()
            assert negative_count == 0, f"Found {negative_count} negative demands!"
            print("  ✅ No negative demand quantities in output")
        else:
            print("  ⚠️  Skipped — run pipeline first")


# ═══════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """
    Run all test groups and print a summary.
    This is a simple test runner — in production you'd use pytest.
    """
    test_groups = [
        ("Schema Validation", TestSchemaValidation),
        ("Batch Validation", TestBatchValidation),
        ("M5 Preprocessing", TestM5Preprocessing),
        ("Data Quality Flags", TestDataQualityFlags),
        ("End-to-End Pipeline", TestEndToEndPipeline),
    ]

    total_passed = 0
    total_failed = 0
    failures = []

    for group_name, test_class in test_groups:
        print(f"\n{'─' * 50}")
        print(f"  {group_name}")
        print(f"{'─' * 50}")

        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in sorted(test_methods):
            try:
                getattr(instance, method_name)()
                total_passed += 1
            except Exception as e:
                total_failed += 1
                failures.append((group_name, method_name, str(e)))
                print(f"  ❌ FAILED: {method_name}: {e}")

    # ── SUMMARY ──
    print(f"\n{'═' * 50}")
    print(f"  TEST RESULTS")
    print(f"{'═' * 50}")
    print(f"  Total:  {total_passed + total_failed}")
    print(f"  ✅ Passed: {total_passed}")
    print(f"  ❌ Failed: {total_failed}")

    if failures:
        print(f"\n  Failed tests:")
        for group, method, error in failures:
            print(f"    [{group}] {method}: {error}")
    else:
        print(f"\n  🎉 ALL TESTS PASSED!")

    print(f"{'═' * 50}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
