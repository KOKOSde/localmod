# LocalMod Deployment Guide

## Table of Contents
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Production Setup](#production-setup)
- [Configuration](#configuration)
- [Monitoring](#monitoring)

## Quick Start

### Local Development

```bash
# Install
pip install -e ".[all]"

# Start server
localmod serve --reload
```

### Docker (Recommended for Production)

```bash
# Build image
docker build -f docker/Dockerfile -t localmod:latest .

# Run container
docker run -d -p 8000:8000 --name localmod localmod:latest
```

## Docker Deployment

### CPU-Only Deployment

```bash
# Using docker-compose
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop
docker-compose -f docker/docker-compose.yml down
```

### GPU Deployment

Requires NVIDIA Docker runtime.

```bash
# Install NVIDIA Container Toolkit
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Build GPU image
docker build -f docker/Dockerfile.gpu -t localmod:gpu .

# Run with GPU
docker run --gpus all -d -p 8000:8000 --name localmod-gpu localmod:gpu

# Or use docker-compose
docker-compose -f docker/docker-compose.yml --profile gpu up -d
```

### Pre-downloading Models

To avoid downloading models on first request:

```bash
# Download models first
python scripts/download_models.py --model-dir /path/to/models

# Run with pre-downloaded models in offline mode
docker run -d -p 8000:8000 \
  -v /path/to/models:/models \
  -e LOCALMOD_MODEL_DIR=/models \
  -e LOCALMOD_OFFLINE=1 \
  -e LOCALMOD_LAZY_LOAD=false \
  localmod:latest
```

### Offline Mode

For air-gapped or security-sensitive deployments:

```bash
# 1. Download models on a machine with internet
python scripts/download_models.py --model-dir /path/to/models

# 2. Copy /path/to/models to your secure environment

# 3. Run in offline mode
export LOCALMOD_MODEL_DIR=/path/to/models
export LOCALMOD_OFFLINE=1
localmod serve
```

## Production Setup

### Environment Variables

Create a `.env` file:

```bash
LOCALMOD_HOST=0.0.0.0
LOCALMOD_PORT=8000
LOCALMOD_DEVICE=cpu
LOCALMOD_LAZY_LOAD=false
LOCALMOD_LOG_LEVEL=INFO
LOCALMOD_WORKERS=4
LOCALMOD_TOXICITY_THRESHOLD=0.5
LOCALMOD_PII_THRESHOLD=0.5
LOCALMOD_SPAM_THRESHOLD=0.5
```

### Running with Multiple Workers

```bash
# Using uvicorn directly
uvicorn localmod.api.app:app --host 0.0.0.0 --port 8000 --workers 4

# Using CLI
localmod serve --workers 4
```

**Note:** With multiple workers, each worker loads its own models. Ensure sufficient memory.

### Reverse Proxy (nginx)

```nginx
upstream localmod {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localmod;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection "";
        proxy_connect_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Systemd Service

Create `/etc/systemd/system/localmod.service`:

```ini
[Unit]
Description=LocalMod Content Moderation API
After=network.target

[Service]
Type=simple
User=localmod
Group=localmod
WorkingDirectory=/opt/localmod
Environment=LOCALMOD_DEVICE=cpu
Environment=LOCALMOD_LAZY_LOAD=false
ExecStart=/usr/local/bin/localmod serve --workers 4
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable localmod
sudo systemctl start localmod
```

## Configuration

### Memory Requirements

| Configuration | RAM | VRAM (GPU) | Disk |
|--------------|-----|------------|------|
| PII only | ~200MB | N/A | 0 |
| Toxicity ensemble | ~4GB | ~6GB | ~2GB |
| All classifiers (CPU) | ~6GB | N/A | ~3GB |
| All classifiers (GPU) | ~2GB | ~8GB | ~3GB |

**Note:** Toxicity detection now uses an ensemble of 4 models for 0.75 balanced accuracy (matching Amazon Comprehend).

### Recommended Settings by Use Case

**High Security (low false negatives):**
```bash
LOCALMOD_TOXICITY_THRESHOLD=0.3
LOCALMOD_PII_THRESHOLD=0.3
LOCALMOD_PROMPT_INJECTION_THRESHOLD=0.3
```

**High Precision (low false positives):**
```bash
LOCALMOD_TOXICITY_THRESHOLD=0.7
LOCALMOD_PII_THRESHOLD=0.5
LOCALMOD_PROMPT_INJECTION_THRESHOLD=0.7
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Docker Health Check

Built into the Docker images:
```bash
docker inspect --format='{{.State.Health.Status}}' localmod
```

### Prometheus Metrics (Future)

Planned for v0.2.0.

## Troubleshooting

### Model Download Failures

If models fail to download:
1. Check internet connectivity
2. Set `HF_HOME` to a writable directory
3. Pre-download models: `localmod download`

### Out of Memory

1. Use fewer classifiers
2. Reduce batch size: `LOCALMOD_MAX_BATCH_SIZE=8`
3. Use CPU instead of GPU for memory-constrained environments

### Slow First Request

First request loads models. Set `LOCALMOD_LAZY_LOAD=false` to load on startup.


