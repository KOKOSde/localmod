# LocalMod

**Fully offline content moderation API** — Self-hosted content safety for teams who can't afford to send data to the cloud.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🔒 **100% Offline** — All models run locally, no external API calls
- ⚡ **Fast** — <100ms latency on CPU, <30ms on GPU
- 🎯 **Comprehensive** — Toxicity, PII, prompt injection, spam, and NSFW detection
- 🐳 **Docker Ready** — Single image deployment
- 🔧 **Configurable** — Adjustable thresholds and classifiers

## Quick Start

### Installation

```bash
pip install localmod
```

### Usage

```python
from localmod import SafetyPipeline

pipeline = SafetyPipeline()
result = pipeline.analyze("Hello, how are you?")
print(result.flagged)  # False
```

### API Server

```bash
localmod serve
# API available at http://localhost:8000
```

### Docker

```bash
docker run -p 8000:8000 localmod:latest
```

## Classifiers

| Classifier | Description |
|------------|-------------|
| **toxicity** | Hate speech, harassment, threats, profanity |
| **pii** | Emails, phones, SSNs, credit cards, addresses |
| **prompt_injection** | LLM jailbreak attempts, instruction override |
| **spam** | Promotional content, scams, repetitive messages |
| **nsfw** | Sexual content, adult themes (text only) |

## API Endpoints

- `POST /analyze` — Analyze single text
- `POST /analyze/batch` — Analyze multiple texts
- `POST /redact` — Redact PII from text
- `GET /health` — Health check
- `GET /classifiers` — List available classifiers

## Configuration

Environment variables (prefix with `LOCALMOD_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `PORT` | `8000` | API server port |
| `LAZY_LOAD` | `true` | Load models on first request |

## Development

```bash
# Install with dev dependencies
pip install -e ".[all]"

# Run tests
make test

# Start dev server
make serve
```

## License

MIT License — see [LICENSE](LICENSE) for details.

