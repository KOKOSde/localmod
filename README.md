# LocalMod

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Fully offline content moderation API** — Self-hosted content safety for teams who can't send data to the cloud.

<p align="center">
  <img src="docs/architecture.svg" alt="LocalMod Architecture" width="800"/>
</p>

<p align="center">
  <img src="docs/examples.svg" alt="LocalMod Examples" width="900"/>
</p>

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/KOKOSde/localmod.git
cd localmod
pip install -e .

# 2. Download ML models (one-time)
python scripts/download_models.py

# 3. Verify offline readiness
localmod verify-models --offline

# 4. Run the server
localmod serve
```

---

## Offline Mode

LocalMod is designed to run **100% offline** after downloading models once.

### Download Models

```bash
# Download all models to default directory (~/.cache/localmod/models)
python scripts/download_models.py

# Or specify a custom directory
python scripts/download_models.py --model-dir /path/to/models
```

### Verify Offline Readiness

```bash
# Verify models load and work correctly
localmod verify-models --offline
```

### Run Offline

```bash
# Set environment variables
export LOCALMOD_MODEL_DIR=~/.cache/localmod/models
export LOCALMOD_OFFLINE=1

# Run server
localmod serve
```

### Docker (Offline)

```bash
# Build image
docker build -f docker/Dockerfile -t localmod:latest .

# Run with pre-downloaded models mounted
docker run -p 8000:8000 \
  -v /path/to/models:/models \
  -e LOCALMOD_MODEL_DIR=/models \
  -e LOCALMOD_OFFLINE=1 \
  localmod:latest
```

---

## Classifiers

| Classifier | Detects | Model | Needs Download? |
|------------|---------|-------|-----------------|
| 🔒 **PII** | Emails, phones, SSNs, credit cards | Regex | ❌ No |
| 🔥 **Toxicity** | Hate speech, harassment, threats | `unitary/toxic-bert` | ✅ Yes |
| ⚡ **Prompt Injection** | LLM jailbreaks, instruction override | `deepset/deberta-v3-base-injection` | ✅ Yes |
| 📧 **Spam** | Promotional content, scams | `mshenoda/roberta-spam` | ✅ Yes |
| 🔞 **NSFW** | Sexual content, adult themes | `michellejieli/NSFW_text_classifier` | ✅ Yes |

---

## API Usage

### Python

```python
from localmod import SafetyPipeline

pipeline = SafetyPipeline()

# Analyze text
report = pipeline.analyze("Hello, how are you?")
print(f"Flagged: {report.flagged}")  # False

# Detect PII (works without model download)
report = pipeline.analyze("My email is john@example.com", classifiers=["pii"])
print(f"Flagged: {report.flagged}")  # True
```

### REST API

```bash
# Start server
localmod serve --port 8000

# Analyze text
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Contact me at john@example.com", "classifiers": ["pii"]}'
```

### API Response

```json
{
  "flagged": true,
  "results": [
    {
      "classifier": "pii",
      "flagged": true,
      "confidence": 1.0,
      "severity": "medium",
      "categories": ["email"],
      "metadata": {"total_count": 1},
      "explanation": ""
    }
  ],
  "summary": "Content flagged for: pii (medium): email",
  "processing_time_ms": 1.5
}
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classifiers` | GET | List classifiers |
| `/analyze` | POST | Analyze text |
| `/analyze/batch` | POST | Batch analysis |
| `/redact` | POST | Redact PII |

---

## Configuration

Environment variables (prefix with `LOCALMOD_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `~/.cache/localmod/models` | Model directory |
| `OFFLINE` | `false` | Strict offline mode |
| `DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `LAZY_LOAD` | `true` | Load models on first use |
| `*_THRESHOLD` | `0.5` | Detection threshold |

---

## CLI Commands

```bash
localmod serve              # Start API server
localmod analyze "text"     # Analyze text
localmod download           # Download models
localmod verify-models      # Verify models work
localmod list               # List classifiers
```

---

## Performance

| Metric | CPU | GPU |
|--------|-----|-----|
| Latency (p95) | <100ms | <30ms |
| Throughput | 20+ req/sec | 100+ req/sec |
| Memory | <2GB RAM | <4GB VRAM |

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v -m "not slow"
```

---

## License

MIT License — see [LICENSE](LICENSE).
