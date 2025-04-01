# Base image
FROM python:3.12-slim-bullseye as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Create a non-root user with a fixed UID
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g 1000 -m appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a separate layer for better caching
COPY requirements.txt . 
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Set ownership to the non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create a volume for the Chroma database
VOLUME /app/chroma_db

# Development stage with hot reloading
FROM base as development
RUN pip install uvicorn[standard] watchfiles
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage with Gunicorn for better performance
FROM base as production
RUN pip install gunicorn uvicorn[standard]
CMD ["gunicorn", "app:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
