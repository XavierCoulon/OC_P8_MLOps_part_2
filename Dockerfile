# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock ./

# Build dependencies
RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    PYTHONPATH=/app \
    HF_HOME=/home/user/.cache/huggingface

# Create user with proper permissions (for HF Spaces)
RUN useradd -m -u 1000 user
RUN chown -R user:user /app

# Create HuggingFace cache directory with proper permissions
RUN mkdir -p /home/user/.cache/huggingface && \
    chown -R user:user /home/user/.cache

# Switch to user
USER user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/api/v1/health')" || exit 1

# Expose port (7860 for HF Spaces)
EXPOSE 7860

# Initialize database and run application
CMD ["sh", "-c", "sleep 5 && python -m app.db.init_db && uvicorn app.main:app --host 0.0.0.0 --port 7860"]
