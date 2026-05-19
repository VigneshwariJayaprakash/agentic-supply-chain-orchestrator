"""
MAIN.PY — The API Server
===========================

WHAT IS FASTAPI?
    Imagine a restaurant. Customers (users/apps) sit at tables and place
    orders (API requests). The waiter (FastAPI) takes the order, brings it
    to the kitchen (our ML models + RAG), and delivers the food (response)
    back to the customer.

    FastAPI is the WAITER — it listens for requests, routes them to the
    right function, and returns the result.

WHAT IS A REST API?
    REST = rules for how computers talk to each other over the internet.
    Each URL is an "endpoint" that does one thing:

    GET  /health     → "Is the server running?" (health check)
    GET  /parts      → "List all parts" (read data)
    POST /query      → "Ask a question" (send data, get answer)
    POST /forecast   → "Predict demand" (run ML model)

    GET = read something, POST = send something

WHAT IS AN ENDPOINT?
    A URL + function pair. When someone visits /health, FastAPI runs
    the health_check() function and returns its result as JSON.

WHY FASTAPI OVER FLASK:
    - Automatic API documentation (Swagger UI at /docs)
    - Built-in request validation with type hints
    - Async support (handles many requests simultaneously)
    - 3-5x faster than Flask
    - Modern Python (type hints, dataclasses)

INTERVIEW TIP:
    "I deployed the forecasting and RAG pipeline as a FastAPI REST service
    with endpoints for health monitoring, data ingestion, demand forecasting,
    and natural language querying. The API includes automatic OpenAPI
    documentation, request validation via Pydantic models, and structured
    error handling. Response time for RAG queries averages under 2 seconds."
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from typing import Optional
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn pydantic")
    sys.exit(1)


# =========================================================================
# APP SETUP
# =========================================================================

app = FastAPI(
    title="Agentic Supply Chain Orchestrator",
    description="AI-powered demand forecasting and supply chain intelligence API",
    version="1.0.0",
)

# CORS middleware — allows the Streamlit dashboard (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track when the server started
START_TIME = datetime.now()


# =========================================================================
# REQUEST/RESPONSE MODELS (Pydantic)
# =========================================================================
# These define EXACTLY what shape the request and response JSON must have.
# FastAPI automatically validates incoming requests against these models.
# If a required field is missing, FastAPI returns a 422 error with details.

class QueryRequest(BaseModel):
    """What the user sends when asking a question."""
    question: str = Field(..., description="Natural language question about supply chain")
    top_k: int = Field(default=3, description="Number of sources to retrieve", ge=1, le=10)

class QueryResponse(BaseModel):
    """What we send back after answering."""
    question: str
    answer: str
    sources: list[str]
    response_time_ms: float

class ForecastRequest(BaseModel):
    """Request for demand forecast."""
    part_number: Optional[str] = Field(default=None, description="Specific part to forecast")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    uptime_seconds: float
    version: str
    components: dict


# =========================================================================
# LAZY-LOADED COMPONENTS
# =========================================================================
# We load ML models and RAG pipeline only when first needed (lazy loading).
# This makes the server start fast — models load on first request.

_rag_pipeline = None
_knowledge_base_loaded = False


def get_rag_pipeline():
    """Load RAG pipeline on first use."""
    global _rag_pipeline, _knowledge_base_loaded

    if _rag_pipeline is not None:
        return _rag_pipeline

    try:
        from src.genai.rag_pipeline import create_rag_pipeline
        _rag_pipeline = create_rag_pipeline()
        _knowledge_base_loaded = True
        return _rag_pipeline
    except Exception as e:
        print(f"Warning: Could not load RAG pipeline: {e}")
        return None


# =========================================================================
# ENDPOINTS
# =========================================================================

@app.get("/", tags=["General"])
def root():
    """Root endpoint — basic info about the API."""
    return {
        "name": "Agentic Supply Chain Orchestrator",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/parts", "/query", "/forecast", "/metrics"],
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check():
    """
    Health check endpoint.

    WHAT: Returns whether the server is running and all components are loaded.
    WHY: Production systems need health checks for monitoring tools (like
    Datadog or AWS CloudWatch) to verify the service is alive.
    WHEN: Called every 30 seconds by the load balancer.
    """
    uptime = (datetime.now() - START_TIME).total_seconds()

    # Check component status
    rag_status = "loaded" if _rag_pipeline is not None else "not loaded (loads on first query)"
    kb_path = "data/processed/knowledge_base.json"
    kb_status = "available" if os.path.exists(kb_path) else "not built"
    model_status = "available" if os.path.exists("models/xgboost_model.pkl") else "not trained"

    return HealthResponse(
        status="healthy",
        uptime_seconds=round(uptime, 1),
        version="1.0.0",
        components={
            "rag_pipeline": rag_status,
            "knowledge_base": kb_status,
            "xgboost_model": model_status,
        },
    )


@app.get("/parts", tags=["Data"])
def list_parts():
    """
    List all parts in the knowledge base.

    WHAT: Returns a list of all semiconductor parts with their metadata.
    WHY: Lets users discover what parts exist before asking specific questions.
    """
    kb_path = "data/processed/knowledge_base.json"
    if not os.path.exists(kb_path):
        raise HTTPException(status_code=404, detail="Knowledge base not built. Run: python -m src.genai.knowledge_base")

    with open(kb_path) as f:
        chunks = json.load(f)

    parts = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        if meta.get("source") == "part_profile":
            parts.append({
                "part_number": meta.get("part_number"),
                "demand_type": meta.get("demand_type"),
                "region": meta.get("region"),
                "trend": meta.get("trend"),
                "stockout_risk": meta.get("stockout_risk"),
            })

    return {"count": len(parts), "parts": parts}


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_supply_chain(request: QueryRequest):
    """
    Ask a natural language question about your supply chain.

    WHAT: Takes a plain English question, searches the knowledge base,
    and returns a data-grounded answer.

    EXAMPLES:
        "Which parts are at stockout risk?"
        "What is the forecast accuracy for NPI parts?"
        "Compare standard vs NPI demand patterns"
        "Tell me about AMAT-RF-7800-GEN"

    HOW IT WORKS:
        1. Embed the question using Sentence Transformers
        2. Search ChromaDB for the most relevant data chunks
        3. Send question + context to GPT-4
        4. Return the grounded answer with sources
    """
    start = time.time()

    pipeline = get_rag_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not available. Check knowledge base and dependencies.",
        )

    result = pipeline.query(request.question, top_k=request.top_k)

    elapsed = (time.time() - start) * 1000  # Convert to milliseconds

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        response_time_ms=round(elapsed, 1),
    )


@app.get("/metrics", tags=["Data"])
def get_model_metrics():
    """
    Get model comparison metrics (XGBoost vs ARIMA vs Naive).

    WHAT: Returns the evaluation results from Phase 2.
    WHY: Lets users and dashboards display forecast accuracy.
    """
    comparison_path = "data/processed/model_comparison.csv"
    if not os.path.exists(comparison_path):
        raise HTTPException(status_code=404, detail="Model comparison not found. Run Phase 2 first.")

    import pandas as pd
    df = pd.read_csv(comparison_path)

    metrics = []
    for _, row in df.iterrows():
        metrics.append({
            "model": row["model"],
            "mae": row["mae"],
            "rmse": row["rmse"],
            "mape": row["mape"],
        })

    return {"models": metrics}


@app.post("/forecast", tags=["ML"])
def run_forecast(request: ForecastRequest):
    """
    Get demand forecast for parts.

    WHAT: Returns predictions from the XGBoost model.
    WHY: Lets other systems programmatically get demand forecasts.
    """
    predictions_path = "data/processed/xgboost_predictions.csv"
    if not os.path.exists(predictions_path):
        raise HTTPException(status_code=404, detail="Predictions not found. Run Phase 2 first.")

    import pandas as pd
    df = pd.read_csv(predictions_path)

    summary = {
        "total_predictions": len(df),
        "mean_actual": round(df["actual"].mean(), 2),
        "mean_predicted": round(df["predicted"].mean(), 2),
        "mae": round((df["actual"] - df["predicted"]).abs().mean(), 2),
    }

    # Return sample predictions
    sample = df.head(10).to_dict(orient="records")

    return {"summary": summary, "sample_predictions": sample}


# =========================================================================
# RUN THE SERVER
# =========================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting Agentic Supply Chain Orchestrator API...")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
