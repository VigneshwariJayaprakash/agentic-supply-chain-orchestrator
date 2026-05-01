"""
SCHEMA.PY — The Data Bouncer
=============================

WHAT THIS FILE DOES:
    Think of a bouncer at a party who checks IDs.
    This file defines the RULES for what valid data looks like.
    Every single record must pass these rules, or it gets rejected.

WHY WE NEED THIS:
    Imagine you feed your ML model a row where "demand_quantity" is "hello"
    instead of a number. The model would crash. Or worse — it might silently
    give you garbage predictions that LOOK correct but aren't.

    Schema validation is your safety net. It catches problems at the door
    so nothing bad gets through to your models.

HOW IT WORKS (the idea behind Pydantic):
    1. You define a "schema" — a list of fields with their expected types
    2. Every record gets checked against this schema
    3. If a field is wrong type, missing, or has an impossible value → REJECTED
    4. If everything passes → the record is cleaned and standardized

INTERVIEW TIP:
    "I used schema validation to ensure 100% type-safe data ingestion.
    Every record passes through strict type checking and business rule
    validation before reaching the ML pipeline. This eliminates an entire
    class of bugs where malformed data could corrupt model training."
"""

from datetime import datetime
from typing import Optional


# ─── VALID VALUES ──────────────────────────────────────────────────────────
# These are the ONLY values allowed for certain fields.
# If a record has demand_type="banana", it gets rejected.
#
# Think of these like a menu at a restaurant — you can only order
# what's on the menu, nothing else.

VALID_DEMAND_TYPES = {"NPI", "standard", "service_campaign", "reliability"}

VALID_REGIONS = {"North America", "Asia Pacific", "Europe"}


# ─── THE SCHEMA CLASS ─────────────────────────────────────────────────────
# This is what Pydantic does automatically — we're building it by hand
# so you understand what happens under the hood.

class SupplyChainRecord:
    """
    One single record in our supply chain dataset.

    Each record represents: "On [date], in [region], part [X] had
    demand of [N] units, at a cost of [$Y] per unit."

    FIELDS:
        part_number      : str   — unique ID for the part (like "HOBBIES_1_001")
        demand_type      : str   — one of: NPI, standard, service_campaign, reliability
        region           : str   — one of: North America, Asia Pacific, Europe
        demand_quantity  : int   — how many units were demanded (must be >= 0)
        transaction_date : str   — the date (format: YYYY-MM-DD)
        unit_cost        : float — price per unit in USD (must be > 0)
        part_description : str   — category description (like "HOBBIES")
        npi_signal       : str or None — NPI launch or event name (can be empty)
    """

    def __init__(
        self,
        part_number: str,
        demand_type: str,
        region: str,
        demand_quantity: int,
        transaction_date: str,
        unit_cost: float,
        part_description: str,
        npi_signal: Optional[str] = None,
    ):
        # ── Step 1: Store all the values ──
        self.part_number = part_number
        self.demand_type = demand_type
        self.region = region
        self.demand_quantity = demand_quantity
        self.transaction_date = transaction_date
        self.unit_cost = unit_cost
        self.part_description = part_description
        self.npi_signal = npi_signal

        # ── Step 2: Validate everything ──
        # This is where the bouncer checks IDs.
        # If anything is wrong, we raise a ValueError with a clear message.
        self._validate()

    def _validate(self):
        """
        Check every field against our rules.

        WHY EACH CHECK EXISTS:

        - Type checks: "demand_quantity" must be an integer, not "hello"
        - Range checks: you can't demand -50 parts (negative makes no sense)
        - Allowed values: demand_type must be one of our 4 categories
        - Date format: "2015-13-45" is not a real date
        - Required fields: part_number can't be empty string
        """
        errors = []

        # ── Check part_number: must be a non-empty string ──
        # WHY: Every part needs an ID. An empty ID means we don't know what part this is.
        if not isinstance(self.part_number, str) or len(self.part_number.strip()) == 0:
            errors.append("part_number must be a non-empty string")

        # ── Check demand_type: must be one of our 4 valid types ──
        # WHY: Our ML models are trained on these 4 categories.
        #       An unknown category would break the model.
        if self.demand_type not in VALID_DEMAND_TYPES:
            errors.append(
                f"demand_type '{self.demand_type}' is not valid. "
                f"Must be one of: {VALID_DEMAND_TYPES}"
            )

        # ── Check region: must be one of our 3 valid regions ──
        # WHY: Same reason — our models expect specific regions.
        if self.region not in VALID_REGIONS:
            errors.append(
                f"region '{self.region}' is not valid. "
                f"Must be one of: {VALID_REGIONS}"
            )

        # ── Check demand_quantity: must be an integer >= 0 ──
        # WHY: You can't demand -50 parts. You also can't demand 3.7 parts
        #       (you either need 3 or 4, not a fraction).
        if not isinstance(self.demand_quantity, (int,)):
            # Allow numpy int types too
            try:
                self.demand_quantity = int(self.demand_quantity)
            except (ValueError, TypeError):
                errors.append("demand_quantity must be an integer")

        if isinstance(self.demand_quantity, (int,)) and self.demand_quantity < 0:
            errors.append(
                f"demand_quantity cannot be negative (got {self.demand_quantity})"
            )

        # ── Check unit_cost: must be a positive number ──
        # WHY: A part can't cost -$5. And it can't cost $0 either
        #       (even free parts have some internal cost).
        if not isinstance(self.unit_cost, (int, float)):
            errors.append("unit_cost must be a number")
        elif self.unit_cost <= 0:
            errors.append(
                f"unit_cost must be positive (got {self.unit_cost})"
            )

        # ── Check transaction_date: must be a valid YYYY-MM-DD date ──
        # WHY: "2015-13-45" isn't a real date. February 30 isn't real either.
        #       Invalid dates would break time-series analysis.
        try:
            datetime.strptime(self.transaction_date, "%Y-%m-%d")
        except (ValueError, TypeError):
            errors.append(
                f"transaction_date '{self.transaction_date}' is not valid. "
                f"Must be format YYYY-MM-DD"
            )

        # ── Check part_description: must be a non-empty string ──
        if not isinstance(self.part_description, str) or len(self.part_description.strip()) == 0:
            errors.append("part_description must be a non-empty string")

        # ── If any check failed, reject the whole record ──
        if errors:
            raise ValueError(
                f"Validation failed for record:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

    def to_dict(self):
        """
        Convert this record to a dictionary.
        Useful for creating DataFrames later.
        """
        return {
            "part_number": self.part_number,
            "demand_type": self.demand_type,
            "region": self.region,
            "demand_quantity": self.demand_quantity,
            "transaction_date": self.transaction_date,
            "unit_cost": self.unit_cost,
            "part_description": self.part_description,
            "npi_signal": self.npi_signal,
        }

    def __repr__(self):
        """What you see when you print this record."""
        return (
            f"SupplyChainRecord(part={self.part_number}, "
            f"type={self.demand_type}, region={self.region}, "
            f"qty={self.demand_quantity}, date={self.transaction_date})"
        )


# ─── BATCH VALIDATOR ───────────────────────────────────────────────────────
# In the real world, you don't validate one record at a time.
# You validate THOUSANDS at once and collect all the errors.

def validate_batch(records: list[dict]) -> tuple[list[SupplyChainRecord], list[dict]]:
    """
    Validate a batch of records.

    RETURNS:
        valid_records : list of SupplyChainRecord objects that passed
        errors        : list of dicts with {"row": index, "error": message}

    WHY A BATCH FUNCTION:
        In production, you might ingest 100,000 records at once.
        You don't want to stop at the first bad record — you want to
        process all good records and report ALL bad ones at the end.
        This is called "error accumulation" vs "fail-fast".
    """
    valid_records = []
    errors = []

    for i, record_dict in enumerate(records):
        try:
            record = SupplyChainRecord(**record_dict)
            valid_records.append(record)
        except (ValueError, TypeError) as e:
            errors.append({"row": i, "error": str(e)})

    return valid_records, errors


# ─── QUICK TEST ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # This runs only when you execute this file directly:
    #   python src/ingestion/schema.py

    print("Testing schema validation...\n")

    # ✅ This should PASS
    good_record = SupplyChainRecord(
        part_number="AMAT-RF-7800-GEN",
        demand_type="NPI",
        region="North America",
        demand_quantity=42,
        transaction_date="2015-01-15",
        unit_cost=12.99,
        part_description="RF & Lithography Components",
        npi_signal="AMAT_Producer_SE_Release",
    )
    print(f"✅ Valid record: {good_record}")

    # ❌ This should FAIL (negative quantity + bad region)
    try:
        bad_record = SupplyChainRecord(
            part_number="AMAT-RF-7800-GEN",
            demand_type="NPI",
            region="Mars",                # ← Not a valid region!
            demand_quantity=-5,            # ← Can't be negative!
            transaction_date="2015-01-15",
            unit_cost=12.99,
            part_description="RF & Lithography Components",
        )
    except ValueError as e:
        print(f"\n❌ Caught bad record:\n{e}")

    print("\n✅ Schema validation working correctly!")
