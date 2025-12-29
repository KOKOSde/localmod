# LocalMod Deployment Guide

## Table of Contents
- [Offline Models](#offline-models)
- [Docker Deployment](#docker-deployment)
- [Production Setup](#production-setup)
- [Configuration](#configuration)

## Offline Models

LocalMod is designed to run **100% offline** after a one-time model download.

### Download Models

```bash
# Download all models to default directory
python scripts/download_models.py

# Or specify custom directory
python scripts/download_models.py --model-dir /opt/localmod/models
```

Models are saved to `~/.cache/localmod/models` by default.

### Verify Models

```bash
localmod verify-models --offline
```

### Configure Offline Mode

```bash
# Environment variables
export LOCALMOD_MODEL_DIR=/path/to/models
export LOCALMOD_OFFLINE=1

# Start server
localmod serve
```

### Manifest

After downloading, a `manifest.json` is created:

```json
{
  "downloaded_at": "2024-01-15T10:30:00",
  "transformers_version": "4.36.0",
  "torch_version": "2.1.0",
  "models": {
    "toxicity": {
      "hf_model_id": "unitary/toxic-bert",
      "local_path": "/path/to/models/toxicity"
    }
  }
}
```

## Docker Deployment

### CPU Deployment

```bash
# Build
docker build -f docker/Dockerfile -t localmod:latest .

# Run
docker run -p 8000:8000 localmod:latest
```

### With Pre-Downloaded Models

```bash
# 1. Download models on host
python scripts/download_models.py --model-dir ./models

# 2. Run container with mounted models
docker run -p 8000:8000 \
  -v $(pwd)/models:/models:ro \
  -e LOCALMOD_MODEL_DIR=/models \
  -e LOCALMOD_OFFLINE=1 \
  localmod:latest
```

### GPU Deployment

```bash
# Build GPU image
docker build -f docker/Dockerfile.gpu -t localmod:gpu .

# Run with GPU
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/models:ro \
  -e LOCALMOD_MODEL_DIR=/models \
  -e LOCALMOD_OFFLINE=1 \
  -e LOCALMOD_DEVICE=cuda \
  localmod:gpu
```

### Docker Compose

```yaml
version: "3.8"
services:
  localmod:
    image: localmod:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models:ro
    environment:
      - LOCALMOD_MODEL_DIR=/models
      - LOCALMOD_OFFLINE=1
      - LOCALMOD_LAZY_LOAD=false
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Production Setup

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCALMOD_MODEL_DIR` | `~/.cache/localmod/models` | Model directory |
| `LOCALMOD_OFFLINE` | `false` | Strict offline mode |
| `LOCALMOD_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `LOCALMOD_LAZY_LOAD` | `true` | Load models on first use |
| `LOCALMOD_WORKERS` | `1` | Number of workers |
| `LOCALMOD_*_THRESHOLD` | `0.5` | Per-classifier threshold |

### Systemd Service

```ini
[Unit]
Description=LocalMod Content Moderation API
After=network.target

[Service]
Type=simple
User=localmod
WorkingDirectory=/opt/localmod
Environment=LOCALMOD_MODEL_DIR=/opt/localmod/models
Environment=LOCALMOD_OFFLINE=1
Environment=LOCALMOD_LAZY_LOAD=false
ExecStart=/usr/local/bin/localmod serve --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

### Reverse Proxy (nginx)

```nginx
upstream localmod {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localmod;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Configuration

### Memory Requirements

| Configuration | RAM |
|--------------|-----|
| PII only | ~200MB |
| All classifiers (CPU) | ~2GB |
| All classifiers (GPU) | ~1GB (+4GB VRAM) |

### Threshold Tuning

**High Security (catch more):**
```bash
LOCALMOD_TOXICITY_THRESHOLD=0.3
LOCALMOD_PROMPT_INJECTION_THRESHOLD=0.3
```

**High Precision (fewer false positives):**
```bash
LOCALMOD_TOXICITY_THRESHOLD=0.7
LOCALMOD_PROMPT_INJECTION_THRESHOLD=0.7
```

## Troubleshooting

### Offline Mode Errors

```
FileNotFoundError: Model for 'toxicity' not found
```

**Fix:** Download models first:
```bash
python scripts/download_models.py
localmod verify-models
```

### Out of Memory

1. Use fewer classifiers
2. Reduce `LOCALMOD_MAX_BATCH_SIZE`
3. Use CPU mode

### Slow First Request

Set `LOCALMOD_LAZY_LOAD=false` to load models at startup.
