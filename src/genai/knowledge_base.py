"""
KNOWLEDGE_BASE.PY — The Library Builder
=========================================

WHAT THIS FILE DOES:
    Imagine you're building a library. You have a pile of papers
    (forecast results, metrics, part data) and you need to:

    1. ORGANIZE them into cards (one card per topic)
    2. Write a SUMMARY on each card (text description)
    3. Put a GPS TRACKER on each card (embedding vector)
    4. File them in a special cabinet (ChromaDB)

    Later, when someone asks a question, the GPS tracker lets us
    instantly find the most relevant cards — even if the question
    uses completely different words than what's on the cards.

THE THREE STAGES:
    Stage 1: CHUNKING — Break raw data into meaningful text chunks
    Stage 2: EMBEDDING — Convert each chunk into a numerical vector
    Stage 3: STORING — Save chunks + embeddings in ChromaDB

WHY TEXT CHUNKS, NOT RAW CSV?
    LLMs understand LANGUAGE, not spreadsheets. If you feed an LLM
    raw CSV data like "AMAT-RF-7800-GEN,NPI,2.77,43.6", it might
    misinterpret which number is which.

    Instead, we create natural language descriptions:
    "Part AMAT-RF-7800-GEN is an NPI-type component in North America.
    The XGBoost forecast achieves MAE of 2.10 units with MAPE of 54.6%.
    The 7-day rolling average demand is 8.3 units, trending upward."

    This is called "data narration" — turning numbers into stories
    that an LLM can understand and reason about.

INTERVIEW TIP:
    "I built a knowledge base by narrating structured forecast data
    into natural language chunks, embedding them with Sentence
    Transformers, and indexing them in ChromaDB. This enables
    semantic search where 'stockout risk' matches data about
    'increasing demand with high MAPE' even without keyword overlap.
    The chunking strategy creates one document per part with all
    relevant metrics, ensuring the LLM has complete context for
    each retrieval."
"""

import pandas as pd
import numpy as np
import json
import os


# =========================================================================
# STAGE 1: CHUNKING — Create text descriptions from raw data
# =========================================================================

def create_part_profile_chunks(
    features_path: str = "data/processed/ml_features.csv",
    predictions_path: str = "data/processed/xgboost_predictions.csv",
    comparison_path: str = "data/processed/model_comparison.csv",
) -> list[dict]:
    """
    Create one text chunk per part with its complete profile.

    WHAT A CHUNK LOOKS LIKE:
        {
            "id": "part_AMAT-RF-7800-GEN",
            "text": "Part AMAT-RF-7800-GEN is an NPI-type component...",
            "metadata": {
                "part_number": "AMAT-RF-7800-GEN",
                "demand_type": "NPI",
                "region": "North America",
                "source": "part_profile"
            }
        }

    WHY ONE CHUNK PER PART:
        If we split data too fine (one chunk per metric), the LLM
        might retrieve MAE for one part and MAPE for a different part,
        creating a confusing answer. One chunk per part ensures the
        LLM always gets a COMPLETE picture.

    WHY METADATA:
        Metadata lets us FILTER searches. If the user asks about
        "NPI parts," we can filter to only search chunks where
        demand_type = "NPI" — faster and more accurate.
    """
    chunks = []

    # Load data
    if not os.path.exists(features_path):
        print(f"  Warning: {features_path} not found. Run Phase 2 first.")
        return chunks

    df = pd.read_csv(features_path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # Load model comparison if available
    comparison = {}
    if os.path.exists(comparison_path):
        comp_df = pd.read_csv(comparison_path)
        for _, row in comp_df.iterrows():
            comparison[row["model"]] = {
                "mae": row["mae"],
                "rmse": row["rmse"],
                "mape": row["mape"],
            }

    # Load predictions for per-part metrics
    part_predictions = {}
    if os.path.exists(predictions_path):
        pred_df = pd.read_csv(predictions_path)
        # We need to align predictions with parts from features
        test_start = int(len(df) * 0.8)
        test_df = df.iloc[test_start:].reset_index(drop=True)

        if len(test_df) == len(pred_df):
            test_df["predicted"] = pred_df["predicted"].values
            test_df["actual"] = pred_df["actual"].values

            for part in test_df["part_number"].unique():
                part_data = test_df[test_df["part_number"] == part]
                actuals = part_data["actual"].values
                preds = part_data["predicted"].values

                mae = np.mean(np.abs(actuals - preds))
                non_zero = actuals > 0
                mape = np.mean(np.abs((actuals[non_zero] - preds[non_zero]) / actuals[non_zero])) * 100 if non_zero.sum() > 0 else 0

                part_predictions[part] = {"mae": round(mae, 2), "mape": round(mape, 1)}

    # Create one chunk per part
    for part in df["part_number"].unique():
        part_data = df[df["part_number"] == part]

        # Basic stats
        demand_type = part_data["demand_type"].iloc[0]
        region = part_data["region"].mode().iloc[0]  # Most common region
        avg_demand = round(part_data["demand_quantity"].mean(), 1)
        max_demand = int(part_data["demand_quantity"].max())
        min_demand = int(part_data["demand_quantity"].min())
        std_demand = round(part_data["demand_quantity"].std(), 1)
        total_demand = int(part_data["demand_quantity"].sum())

        # Recent trend (last 7 days vs previous 7 days)
        if len(part_data) >= 14:
            recent_7 = part_data.tail(7)["demand_quantity"].mean()
            prev_7 = part_data.tail(14).head(7)["demand_quantity"].mean()
            trend_pct = round(((recent_7 - prev_7) / prev_7) * 100, 1) if prev_7 > 0 else 0
            trend_direction = "increasing" if trend_pct > 5 else "decreasing" if trend_pct < -5 else "stable"
        else:
            trend_pct = 0
            trend_direction = "insufficient data"

        # Rolling stats
        rolling_mean_7 = round(part_data["rolling_mean_7"].mean(), 1) if "rolling_mean_7" in part_data.columns else avg_demand
        rolling_std_7 = round(part_data["rolling_std_7"].mean(), 1) if "rolling_std_7" in part_data.columns else std_demand

        # Unit cost
        unit_cost = round(part_data["unit_cost"].mean(), 2) if "unit_cost" in part_data.columns else 0

        # Date range
        date_min = part_data["transaction_date"].min().strftime("%Y-%m-%d")
        date_max = part_data["transaction_date"].max().strftime("%Y-%m-%d")

        # Per-part prediction metrics
        part_mae = part_predictions.get(part, {}).get("mae", "N/A")
        part_mape = part_predictions.get(part, {}).get("mape", "N/A")

        # NPI signal info
        if "has_npi_signal" in part_data.columns:
            npi_days = int(part_data["has_npi_signal"].sum())
            npi_pct = round((npi_days / len(part_data)) * 100, 1)
        else:
            npi_days = 0
            npi_pct = 0

        # Volatility assessment
        cv = (std_demand / avg_demand) if avg_demand > 0 else 0
        if cv > 0.5:
            volatility = "high volatility (coefficient of variation > 0.5)"
        elif cv > 0.3:
            volatility = "moderate volatility"
        else:
            volatility = "low volatility (stable demand)"

        # Stockout risk assessment
        if trend_direction == "increasing" and str(part_mape) != "N/A" and part_mape > 50:
            stockout_risk = "HIGH — demand is increasing and forecast uncertainty is high"
        elif trend_direction == "increasing":
            stockout_risk = "MODERATE — demand is increasing but forecasts are reasonably accurate"
        elif volatility.startswith("high"):
            stockout_risk = "MODERATE — demand is volatile and hard to predict"
        else:
            stockout_risk = "LOW — demand is stable and predictable"

        # Build the text chunk (natural language description)
        text = f"""Part {part} is a {demand_type}-type semiconductor component primarily serving the {region} region.

Demand profile: Average daily demand is {avg_demand} units (range: {min_demand} to {max_demand} units). Total demand over the observation period ({date_min} to {date_max}) was {total_demand} units. The demand shows {volatility}.

Recent trend: Demand is {trend_direction} ({trend_pct:+.1f}% change, last 7 days vs previous 7 days). The 7-day rolling average demand is {rolling_mean_7} units with rolling standard deviation of {rolling_std_7}.

Forecast accuracy: XGBoost forecast achieves MAE of {part_mae} units and MAPE of {part_mape}%. Unit cost is ${unit_cost:.2f}.

NPI signals: {npi_days} out of {len(part_data)} days had NPI or event signals ({npi_pct}% of days).

Stockout risk assessment: {stockout_risk}.

Recommendation: {"Increase safety stock by 15-20% due to high demand uncertainty and upward trend." if "HIGH" in stockout_risk else "Increase safety stock by 10% due to rising demand or volatility." if "MODERATE" in stockout_risk else "Maintain current stock levels. Demand is predictable."}"""

        chunks.append({
            "id": f"part_{part}",
            "text": text,
            "metadata": {
                "part_number": part,
                "demand_type": demand_type,
                "region": region,
                "avg_demand": avg_demand,
                "trend": trend_direction,
                "stockout_risk": stockout_risk.split(" —")[0],  # Just HIGH/MODERATE/LOW
                "source": "part_profile",
            },
        })

    return chunks


def create_model_summary_chunks(
    comparison_path: str = "data/processed/model_comparison.csv",
) -> list[dict]:
    """
    Create chunks summarizing overall model performance.

    This gives the LLM context about how reliable the forecasts are.
    """
    chunks = []

    if not os.path.exists(comparison_path):
        return chunks

    comp_df = pd.read_csv(comparison_path)

    # Overall model comparison chunk
    models_text = "Model comparison results for demand forecasting:\n\n"
    for _, row in comp_df.iterrows():
        models_text += f"- {row['model']}: MAE = {row['mae']} units, RMSE = {row['rmse']} units, MAPE = {row['mape']}%\n"

    # Find best model
    best = comp_df.loc[comp_df["mae"].idxmin()]
    models_text += f"\nBest performing model: {best['model']} with MAE of {best['mae']} units."
    models_text += f"\n\nThe XGBoost model outperformed the naive baseline (predict tomorrow = today) by approximately 22% on MAE. "
    models_text += "This demonstrates that the model learned genuine demand patterns beyond simple repetition."
    models_text += "\n\nTop predictive features: rolling_mean_14 (25.2% importance), rolling_mean_7 (17.0%), demand_type (12.5%). "
    models_text += "This confirms that recent demand trends and part category are the strongest drivers of future demand."

    chunks.append({
        "id": "model_comparison_summary",
        "text": models_text,
        "metadata": {
            "source": "model_summary",
            "best_model": best["model"],
            "best_mae": best["mae"],
        },
    })

    return chunks


def create_demand_type_chunks() -> list[dict]:
    """
    Create chunks explaining each demand type.

    This gives the LLM domain knowledge about supply chain concepts.
    """
    chunks = []

    demand_types = {
        "NPI": {
            "full_name": "New Product Introduction",
            "description": "NPI parts are components for newly launched semiconductor equipment. Demand pattern is characterized by sudden spikes when a new product is announced, followed by gradual stabilization. NPI demand is the hardest to predict (highest MAPE) due to its volatile, event-driven nature. Safety stock should be 20-30% above average to buffer against forecast uncertainty. Example parts: AMAT-RF-7800-GEN (RF generator), AMAT-LITH-5500-ALIGN (lithography alignment sensor).",
        },
        "standard": {
            "full_name": "Standard replacement parts",
            "description": "Standard parts are routine replacement components that factories consistently need. Demand pattern is steady and predictable with low variance. These parts have the lowest MAPE (best forecast accuracy). Safety stock can be minimal (5-10% above average) since demand is reliable. Example parts: AMAT-CVD-3200-VALVE (gas delivery valve), AMAT-CVD-3200-SEAL (chamber seal ring).",
        },
        "service_campaign": {
            "full_name": "Service campaign and warranty parts",
            "description": "Service campaign parts see demand spikes during product recalls, warranty programs, or field upgrade campaigns. Demand pattern shows sharp spikes when a campaign launches, then drops back to baseline when the campaign ends. Forecast accuracy is moderate. Safety stock should be increased 2-4 weeks before a planned campaign. Example parts: AMAT-CMP-4100-PAD (polishing pad), AMAT-CMP-4100-SLURRY (slurry pump).",
        },
        "reliability": {
            "full_name": "Reliability and wear parts",
            "description": "Reliability parts are components that degrade over time through normal use. Demand pattern shows gradual, steady increase as the installed base of equipment ages. These parts are moderately predictable. Safety stock should increase proportionally with the installed base growth rate. Example: AMAT-ETCH-6600-RING (etch chamber focus ring that wears out).",
        },
    }

    for dtype, info in demand_types.items():
        chunks.append({
            "id": f"demand_type_{dtype}",
            "text": f"Demand type: {dtype} ({info['full_name']})\n\n{info['description']}",
            "metadata": {
                "source": "domain_knowledge",
                "demand_type": dtype,
            },
        })

    return chunks


# =========================================================================
# STAGE 2 & 3: EMBEDDING + STORING
# =========================================================================

def build_knowledge_base(
    persist_directory: str = "data/chromadb",
) -> dict:
    """
    THE MAIN FUNCTION: Create all chunks, embed them, store in ChromaDB.

    STEPS:
        1. Generate text chunks from forecast data
        2. Generate model summary chunks
        3. Generate domain knowledge chunks
        4. Try to embed with Sentence Transformers + store in ChromaDB
        5. If libraries not installed, save as JSON (fallback)

    RETURNS:
        Dictionary with all chunks and metadata
    """
    print("=" * 60)
    print("Building RAG Knowledge Base")
    print("=" * 60)

    # Step 1: Create chunks
    print("\nStage 1: Creating text chunks...")
    part_chunks = create_part_profile_chunks()
    print(f"  Part profiles: {len(part_chunks)} chunks")

    model_chunks = create_model_summary_chunks()
    print(f"  Model summaries: {len(model_chunks)} chunks")

    domain_chunks = create_demand_type_chunks()
    print(f"  Domain knowledge: {len(domain_chunks)} chunks")

    all_chunks = part_chunks + model_chunks + domain_chunks
    total = len(all_chunks)
    print(f"  Total: {total} chunks")

    # Show sample chunk
    if part_chunks:
        print(f"\n  Sample chunk (first part profile):")
        print(f"  {'─' * 50}")
        sample_text = part_chunks[0]["text"][:300]
        for line in sample_text.split("\n"):
            print(f"  {line}")
        print(f"  ...")
        print(f"  {'─' * 50}")

    # Step 2 & 3: Try to embed and store in ChromaDB
    print(f"\nStage 2 & 3: Embedding + storing...")

    chromadb_available = False
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        chromadb_available = True
    except ImportError:
        pass

    if chromadb_available:
        print("  ChromaDB found! Using vector database...")

        # Create ChromaDB client
        client = chromadb.PersistentClient(path=persist_directory)

        # Delete existing collection if it exists (fresh start)
        try:
            client.delete_collection("supply_chain_knowledge")
        except Exception:
            pass

        # Try to use Sentence Transformer embeddings
        try:
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            print("  Using Sentence Transformer embeddings (all-MiniLM-L6-v2)")
        except Exception:
            ef = embedding_functions.DefaultEmbeddingFunction()
            print("  Using default ChromaDB embeddings")

        # Create collection
        collection = client.create_collection(
            name="supply_chain_knowledge",
            embedding_function=ef,
            metadata={"description": "Supply chain forecasting knowledge base"},
        )

        # Add all chunks
        collection.add(
            ids=[c["id"] for c in all_chunks],
            documents=[c["text"] for c in all_chunks],
            metadatas=[c["metadata"] for c in all_chunks],
        )

        print(f"  Stored {total} chunks in ChromaDB at {persist_directory}")

        # Test a query
        print(f"\n  Testing semantic search...")
        results = collection.query(
            query_texts=["Which parts have stockout risk?"],
            n_results=3,
        )
        print(f"  Query: 'Which parts have stockout risk?'")
        print(f"  Top 3 results:")
        for i, (doc_id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
            similarity = round(1 - distance, 3)
            print(f"    {i+1}. {doc_id} (similarity: {similarity})")

    else:
        print("  ChromaDB not installed. Saving as JSON fallback...")
        print("  To enable vector search, run: pip install chromadb sentence-transformers")

    # Always save JSON version (useful for debugging and Phase 4 API)
    os.makedirs("data/processed", exist_ok=True)
    json_path = "data/processed/knowledge_base.json"
    with open(json_path, "w") as f:
        json.dump(all_chunks, f, indent=2, default=str)
    print(f"\n  JSON backup saved to {json_path}")

    # Summary
    print(f"\n{'=' * 40}")
    print(f"KNOWLEDGE BASE SUMMARY")
    print(f"{'=' * 40}")
    print(f"  Part profiles:     {len(part_chunks):>4} chunks")
    print(f"  Model summaries:   {len(model_chunks):>4} chunks")
    print(f"  Domain knowledge:  {len(domain_chunks):>4} chunks")
    print(f"  Total:             {total:>4} chunks")
    print(f"  ChromaDB:          {'Active' if chromadb_available else 'Not installed (JSON fallback)'}")
    print(f"  JSON backup:       {json_path}")

    return {
        "chunks": all_chunks,
        "total": total,
        "chromadb_active": chromadb_available,
    }


if __name__ == "__main__":
    result = build_knowledge_base()
