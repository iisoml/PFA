# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Install system dependencies first (for build tools, curl, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only dependency file first (for Docker layer caching)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 6. Copy the entire project into the image
COPY . .

# 7. Create necessary directories (safe fallback)
RUN mkdir -p /app/models /app/artifacts /app/data/processed

# 8. REMOVED: redundant COPYs with invalid shell syntax

# 9. Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/models/xgb_regressor.pkl \
    ARTIFACTS_DIR=/app/artifacts

# 10. Expose FastAPI port
EXPOSE 8000

# 11. Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 12. Run the FastAPI app
ENV PYTHONPATH=/app/src
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]