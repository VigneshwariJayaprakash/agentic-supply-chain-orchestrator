# ================================================================
# DOCKERFILE - Packages everything into a container
# ================================================================
#
# WHAT IS DOCKER? (like you're 10)
#   Imagine you built an amazing science project at home.
#   You want to bring it to school, but it needs your special desk,
#   your lamp, your magnifying glass, and your exact setup.
#
#   Docker lets you put your ENTIRE DESK (code + libraries + settings)
#   into a magic box. When you open the box at school, everything
#   is exactly like it was at home. No "but it works on MY computer!"
#
# HOW TO USE:
#   docker build -t supply-chain-orchestrator .
#   docker run -p 8000:8000 -p 8501:8501 supply-chain-orchestrator
#
# WHAT EACH LINE DOES: see comments below

# Start from a Python base image
# "slim" = smaller image (no extra tools we don't need)
FROM python:3.11-slim

# Set the working directory inside the container
# All commands after this run from /app
WORKDIR /app

# Copy requirements first (Docker caches this layer)
# If requirements.txt hasn't changed, Docker skips reinstalling
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Run the data pipeline and build knowledge base
# These create the processed data files needed at runtime
RUN python data/raw/create_sample_data.py && \
    python -m src.ingestion.pipeline && \
    python -m src.forecasting.feature_engineering && \
    python -m src.forecasting.xgboost_model && \
    python -m src.forecasting.arima_model && \
    python -m src.forecasting.evaluator && \
    python -m src.genai.knowledge_base

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Start both services using a shell script
# FastAPI runs in background, Streamlit in foreground
CMD uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & \
    streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
