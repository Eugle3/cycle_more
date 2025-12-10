# FastAPI Backend Dockerfile for Cloud Run
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer)
COPY api_faf/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api_faf/app/ ./app/

# Copy FAF module (needed for GPX processing)
COPY FAF/ ./FAF/

# Set Python path so imports work
ENV PYTHONPATH=/app:$PYTHONPATH

# Cloud Run will provide PORT environment variable (default 8080)
ENV PORT=8080
EXPOSE 8080

# Run FastAPI with uvicorn
# --host 0.0.0.0 allows external connections (required for Cloud Run)
# --port $PORT uses Cloud Run's assigned port
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
