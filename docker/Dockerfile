# LocalMod CPU Dockerfile
# Multi-stage build for smaller image size

# Build stage
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies (no version pinning for system packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package files - including README.md which is referenced in pyproject.toml
COPY pyproject.toml .
COPY README.md .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Runtime stage
FROM python:3.9-slim AS runtime

WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash localmod && \
    mkdir -p /home/localmod/.cache/localmod && \
    chown -R localmod:localmod /home/localmod

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/localmod /app/localmod

# Set Python path
ENV PYTHONPATH=/app

# Set environment variables
ENV LOCALMOD_HOST=0.0.0.0
ENV LOCALMOD_PORT=8000
ENV LOCALMOD_DEVICE=cpu
ENV LOCALMOD_LAZY_LOAD=true
ENV LOCALMOD_LOG_LEVEL=INFO

# Switch to non-root user
USER localmod

# Set cache directory
ENV HF_HOME=/home/localmod/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/localmod/.cache/huggingface

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "-m", "localmod.cli", "serve"]
