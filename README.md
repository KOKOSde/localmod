# LocalMod

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-109%20passed-brightgreen.svg)]()

**Fully offline content moderation API** — Free, self-hosted, and private. Your data never leaves your infrastructure.

<p align="center">
  <img src="docs/architecture.svg" alt="LocalMod Architecture" width="800"/>
</p>

<p align="center">
  <img src="docs/examples.svg" alt="LocalMod Examples" width="900"/>
</p>

---

## Benchmark Results

### Toxicity Detection

Benchmarked using [CHI 2025 "Lost in Moderation"](https://arxiv.org/html/2503.01623) methodology (HateXplain, Civil Comments, SBIC datasets):

| System | Balanced Accuracy | Type |
|--------|------------------|------|
| OpenAI Moderation API | 0.83 | Commercial |
| Azure Content Moderator | 0.81 | Commercial |
| **LocalMod** | **0.75** ⭐ | Open Source |
| Amazon Comprehend | 0.74 | Commercial |
| Perspective API | 0.62 | Commercial |

### Spam Detection

| System | Balanced Accuracy | Dataset |
|--------|------------------|---------|
| **LocalMod** | **0.998** | [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) |

---

## Installation

```bash
git clone https://github.com/KOKOSde/localmod.git
cd localmod
pip install -e .

# Download ML models (~3.5GB - includes image model)
python scripts/download_models.py
```

## Quick Start

```bash
# Run demo
python examples/demo.py
```

### Python Usage

```python
from localmod import SafetyPipeline

pipeline = SafetyPipeline()
report = pipeline.analyze("Check this text for safety issues")

print(f"Flagged: {report.flagged}")
print(f"Severity: {report.severity}")
```

### API Server

```bash
localmod serve --port 8000

# Test
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "classifiers": ["toxicity", "pii"]}'
```

### Docker

```bash
docker build -f docker/Dockerfile -t localmod:latest .
docker run -p 8000:8000 localmod:latest
```

### Discord Bot 🆕

```bash
# Install Discord dependency
pip install -e ".[discord]"

# Set your bot token
export DISCORD_BOT_TOKEN=your_token_here

# Run the bot
python examples/discord_bot.py
```

Features: Real-time text & image moderation, auto-delete, timeout, logging.

---

## Classifiers

### Text Moderation

| Classifier | Detects | Model |
|------------|---------|-------|
| **PII** | Emails, phones, SSNs, credit cards | Regex + Validation |
| **Toxicity** | Hate speech, harassment, threats | Weighted Ensemble (4 models) |
| **Prompt Injection** | LLM jailbreaks, instruction override | DeBERTa |
| **Spam** | Promotional content, scams | RoBERTa |
| **NSFW Text** | Sexual content, adult themes | NSFW Classifier |

### Image Moderation 🆕

| Classifier | Detects | Model |
|------------|---------|-------|
| **NSFW Image** | Explicit/adult images | [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) (ViT) |

*71M+ downloads on HuggingFace • Apache 2.0 license*

### Toxicity Ensemble

| Model | Weight | Purpose |
|-------|--------|---------|
| [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert) | 50% | Multi-label toxicity |
| [`Hate-speech-CNERG/dehatebert-mono-english`](https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english) | 20% | Hate speech |
| [`s-nlp/roberta_toxicity_classifier`](https://huggingface.co/s-nlp/roberta_toxicity_classifier) | 15% | Toxicity |
| [`facebook/roberta-hate-speech-dynabench-r4-target`](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) | 15% | Adversarial robustness |

---

## API Endpoints

### Text Moderation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze single text |
| `/analyze/batch` | POST | Analyze multiple texts |
| `/redact` | POST | Redact PII from text |

### Image Moderation 🆕

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze/image` | POST | Analyze image from URL |
| `/analyze/image/upload` | POST | Analyze uploaded image file |

### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classifiers` | GET | List text classifiers |
| `/classifiers/image` | GET | List image classifiers |

### Example: Text Analysis

```json
POST /analyze
{
  "text": "You are an idiot!",
  "classifiers": ["toxicity"]
}

Response:
{
  "flagged": true,
  "results": [{"classifier": "toxicity", "flagged": true, "confidence": 0.72, "severity": "high"}],
  "processing_time_ms": 85.3
}
```

### Example: Image Analysis 🆕

```json
POST /analyze/image
{
  "image_url": "https://example.com/image.jpg"
}

Response:
{
  "flagged": false,
  "results": [{"classifier": "nsfw_image", "flagged": false, "confidence": 0.02, "severity": "none"}],
  "processing_time_ms": 120.5
}
```

---

## Offline Mode

```bash
# Download models once
python scripts/download_models.py --model-dir /path/to/models

# Run offline
export LOCALMOD_MODEL_DIR=/path/to/models
export LOCALMOD_OFFLINE=1
localmod serve
```

Docker:
```bash
docker run -p 8000:8000 \
  -v /path/to/models:/models \
  -e LOCALMOD_MODEL_DIR=/models \
  -e LOCALMOD_OFFLINE=1 \
  localmod:latest
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCALMOD_MODEL_DIR` | `~/.cache/localmod/models` | Model directory |
| `LOCALMOD_OFFLINE` | `false` | Force offline mode |
| `LOCALMOD_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |

---

## Performance

| Classifier | CPU Latency | GPU Latency | Memory |
|------------|-------------|-------------|--------|
| PII (regex) | <1ms | <1ms | Minimal |
| Single ML model (text) | ~50-200ms | ~10-30ms | ~1GB |
| Toxicity ensemble (4 models) | ~200-500ms | ~30-80ms | ~3GB |
| NSFW Image (ViT) | ~100-300ms | ~20-50ms | ~500MB |

*Performance varies by hardware. GPU recommended for production workloads.*

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

Models from [HuggingFace](https://huggingface.co/). Benchmark methodology from CHI 2025 "Lost in Moderation" (Hartmann et al.).
