"""
STREAMLIT_APP.PY — The Visual Dashboard
==========================================

WHAT IS STREAMLIT? (like you're 10)
    Imagine you built an amazing LEGO spaceship (your ML models + RAG).
    But your friend can't play with it because they don't know how
    LEGO works. Streamlit puts your spaceship inside a glass display
    case with BUTTONS your friend can press:
    - Button 1: See the spaceship fly (view forecasts)
    - Button 2: Ask questions about it (RAG queries)
    - Button 3: See the stats (model metrics)

    Streamlit turns your Python code into a web page with buttons,
    charts, and text boxes — no HTML/CSS/JavaScript needed.

WHY STREAMLIT OVER REACT/FLASK:
    - 100% Python (no frontend skills needed)
    - Live reload (change code, page updates automatically)
    - Built-in charts, tables, and widgets
    - Perfect for data science dashboards
    - Deploy to Streamlit Cloud for free

HOW TO RUN:
    streamlit run streamlit_app.py

INTERVIEW TIP:
    "I built an interactive Streamlit dashboard that provides three views:
    a KPI overview with model accuracy metrics, a natural language
    query interface powered by our RAG pipeline, and a parts explorer
    with demand type filtering. The dashboard connects to the same
    knowledge base and ML models used by the FastAPI endpoints,
    ensuring consistency across interfaces."
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

import json
import pandas as pd

from dotenv import load_dotenv
load_dotenv()


# =========================================================================
# PAGE CONFIG
# =========================================================================

st.set_page_config(
    page_title="Supply Chain Orchestrator",
    page_icon="🏭",
    layout="wide",
)

st.title("Agentic Supply Chain Orchestrator")
st.caption("AI-powered demand forecasting and supply chain intelligence")


# =========================================================================
# SIDEBAR
# =========================================================================

st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select a view",
    ["Dashboard", "Ask a question", "Parts explorer", "Model metrics"],
)


# =========================================================================
# HELPER: Load data
# =========================================================================

@st.cache_data
def load_knowledge_base():
    """Load knowledge base chunks."""
    path = "data/processed/knowledge_base.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

@st.cache_data
def load_model_comparison():
    """Load model comparison metrics."""
    path = "data/processed/model_comparison.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_predictions():
    """Load XGBoost predictions."""
    path = "data/processed/xgboost_predictions.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_features():
    """Load feature-engineered data."""
    path = "data/processed/ml_features.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# =========================================================================
# PAGE 1: DASHBOARD
# =========================================================================

if page == "Dashboard":
    st.header("Supply chain overview")

    # Load data
    kb = load_knowledge_base()
    comparison = load_model_comparison()
    predictions = load_predictions()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    parts_count = len([c for c in kb if c.get("metadata", {}).get("source") == "part_profile"])
    with col1:
        st.metric("Total parts", parts_count)

    if not comparison.empty:
        best_mae = comparison.loc[comparison["mae"].idxmin()]
        with col2:
            st.metric("Best model", best_mae["model"])
        with col3:
            st.metric("Best MAE", f"{best_mae['mae']} units")
        with col4:
            st.metric("Best MAPE", f"{best_mae['mape']}%")

    st.divider()

    # Part risk overview
    st.subheader("Part risk overview")

    high_risk = []
    moderate_risk = []
    low_risk = []

    for chunk in kb:
        meta = chunk.get("metadata", {})
        if meta.get("source") == "part_profile":
            risk = meta.get("stockout_risk", "UNKNOWN")
            entry = {
                "Part": meta.get("part_number", "Unknown"),
                "Type": meta.get("demand_type", "Unknown"),
                "Region": meta.get("region", "Unknown"),
                "Trend": meta.get("trend", "Unknown"),
                "Risk": risk,
            }
            if risk == "HIGH":
                high_risk.append(entry)
            elif risk == "MODERATE":
                moderate_risk.append(entry)
            else:
                low_risk.append(entry)

    if high_risk:
        st.error(f"**{len(high_risk)} parts at HIGH risk**")
        st.dataframe(pd.DataFrame(high_risk), use_container_width=True, hide_index=True)

    if moderate_risk:
        st.warning(f"**{len(moderate_risk)} parts at MODERATE risk**")
        st.dataframe(pd.DataFrame(moderate_risk), use_container_width=True, hide_index=True)

    if low_risk:
        st.success(f"**{len(low_risk)} parts at LOW risk**")
        st.dataframe(pd.DataFrame(low_risk), use_container_width=True, hide_index=True)

    # Predictions chart
    if not predictions.empty:
        st.divider()
        st.subheader("Forecast vs actual demand (test set)")

        chart_data = predictions.head(50).copy()
        chart_data["index"] = range(len(chart_data))
        chart_data = chart_data.rename(columns={"actual": "Actual", "predicted": "Predicted"})

        st.line_chart(chart_data.set_index("index")[["Actual", "Predicted"]])


# =========================================================================
# PAGE 2: ASK A QUESTION (RAG)
# =========================================================================

elif page == "Ask a question":
    st.header("Ask a supply chain question")
    st.caption("Powered by RAG — answers are grounded in your forecast data")

    # Sample questions
    st.markdown("**Try one of these:**")
    sample_cols = st.columns(3)
    with sample_cols[0]:
        if st.button("Which parts are at stockout risk?"):
            st.session_state["query_input"] = "Which parts are at stockout risk?"
    with sample_cols[1]:
        if st.button("Forecast accuracy for NPI parts?"):
            st.session_state["query_input"] = "What is the forecast accuracy for NPI parts?"
    with sample_cols[2]:
        if st.button("Compare standard vs NPI"):
            st.session_state["query_input"] = "Compare standard vs NPI demand patterns"

    # Query input
    default_query = st.session_state.get("query_input", "")
    question = st.text_input("Your question:", value=default_query, placeholder="e.g., Which parts need more safety stock?")

    if question:
        with st.spinner("Searching knowledge base and generating answer..."):
            try:
                from src.genai.rag_pipeline import create_rag_pipeline

                if "rag_pipeline" not in st.session_state:
                    st.session_state["rag_pipeline"] = create_rag_pipeline()

                pipeline = st.session_state["rag_pipeline"]
                result = pipeline.query(question)

                st.subheader("Answer")
                st.write(result["answer"])

                with st.expander("Sources used"):
                    for source in result["sources"]:
                        st.code(source)

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure you've run: python -m src.genai.knowledge_base")


# =========================================================================
# PAGE 3: PARTS EXPLORER
# =========================================================================

elif page == "Parts explorer":
    st.header("Parts explorer")

    kb = load_knowledge_base()
    features = load_features()

    # Filter by demand type
    demand_types = ["All"] + sorted(set(
        c.get("metadata", {}).get("demand_type", "")
        for c in kb
        if c.get("metadata", {}).get("source") == "part_profile"
    ))
    selected_type = st.selectbox("Filter by demand type:", demand_types)

    # Show part profiles
    for chunk in kb:
        meta = chunk.get("metadata", {})
        if meta.get("source") != "part_profile":
            continue
        if selected_type != "All" and meta.get("demand_type") != selected_type:
            continue

        part = meta.get("part_number", "Unknown")
        risk = meta.get("stockout_risk", "UNKNOWN")

        risk_color = "red" if risk == "HIGH" else "orange" if risk == "MODERATE" else "green"

        with st.expander(f"**{part}** — {meta.get('demand_type', '')} | Risk: :{risk_color}[{risk}]"):
            st.markdown(chunk["text"])

    # Demand distribution chart
    if not features.empty and "demand_type" in features.columns:
        st.divider()
        st.subheader("Demand distribution by type")
        type_stats = features.groupby("demand_type")["demand_quantity"].agg(["mean", "std", "sum"]).round(2)
        type_stats.columns = ["Avg daily demand", "Std deviation", "Total demand"]
        st.dataframe(type_stats, use_container_width=True)


# =========================================================================
# PAGE 4: MODEL METRICS
# =========================================================================

elif page == "Model metrics":
    st.header("Model performance comparison")

    comparison = load_model_comparison()

    if comparison.empty:
        st.warning("No model comparison data found. Run Phase 2 first.")
    else:
        # Metrics table
        st.dataframe(
            comparison.style.highlight_min(subset=["mae", "rmse", "mape"], color="lightgreen"),
            use_container_width=True,
            hide_index=True,
        )

        # Bar charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("MAE by model")
            st.bar_chart(comparison.set_index("model")["mae"])
        with col2:
            st.subheader("MAPE by model")
            st.bar_chart(comparison.set_index("model")["mape"])

        # Feature importance
        st.divider()
        st.subheader("Top predictive features")
        st.markdown("""
        Based on XGBoost feature importance analysis:

        | Rank | Feature | Importance |
        |------|---------|-----------|
        | 1 | rolling_mean_14 | 25.2% |
        | 2 | rolling_mean_7 | 17.0% |
        | 3 | demand_type_encoded | 12.5% |
        | 4 | rolling_std_7 | 7.3% |
        | 5 | unit_cost | 5.2% |
        """)


# =========================================================================
# FOOTER
# =========================================================================

st.sidebar.divider()
st.sidebar.caption("Built with FastAPI + Streamlit + ChromaDB + XGBoost")
st.sidebar.caption("Agentic Supply Chain Orchestrator v1.0.0")
