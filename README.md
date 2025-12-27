# LocalMod

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-91%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-74%25-green.svg)]()

**Fully offline content moderation API** — Self-hosted content safety for teams who can't afford to send data to the cloud.

## Why LocalMod?

- 🔒 **100% Offline** — All models run locally, no external API calls
- ⚡ **Fast** — <100ms latency on CPU, <30ms on GPU
- 🎯 **Comprehensive** — Toxicity, PII, prompt injection, spam, and NSFW detection
- 🐳 **Docker Ready** — Single image deployment
- 🔧 **Configurable** — Adjustable thresholds per classifier
- 💰 **Cost Effective** — No per-request API fees

## Quick Start

### Installation

```bash
pip install -e .
```

### Python Usage

```python
from localmod import SafetyPipeline

# Create pipeline
pipeline = SafetyPipeline()

# Analyze text
report = pipeline.analyze("Hello, how are you?")
print(f"Flagged: {report.flagged}")  # False
print(f"Summary: {report.summary}")   # Content passed all safety checks.

# Analyze text with PII
report = pipeline.analyze("My email is john@example.com")
print(f"Flagged: {report.flagged}")  # True
print(f"Severity: {report.severity}")  # medium

# Use specific classifiers
report = pipeline.analyze(
    "Some text",
    classifiers=["pii", "toxicity"],
    include_explanation=True
)

# Batch analysis
reports = pipeline.analyze_batch([
    "Hello world",
    "Contact: test@example.com",
    "Nice weather today"
])
```

### API Server

```bash
# Start the server
localmod serve

# Or with custom settings
localmod serve --host 0.0.0.0 --port 8000 --reload
```

API available at `http://localhost:8000`

### Docker

```bash
# Build
docker build -f docker/Dockerfile -t localmod:latest .

# Run
docker run -p 8000:8000 localmod:latest

# Or use docker-compose
docker-compose -f docker/docker-compose.yml up -d
```

## Classifiers

| Classifier | Description | Model Type |
|------------|-------------|------------|
| **toxicity** | Hate speech, harassment, threats, profanity | ML (DistilBERT) |
| **pii** | Emails, phones, SSNs, credit cards, IPs | Regex + Validation |
| **prompt_injection** | LLM jailbreaks, instruction override | Pattern + ML |
| **spam** | Promotional content, scams, click-bait | Heuristics + ML |
| **nsfw** | Sexual content, adult themes (text only) | ML |

### PII Detection Details

The PII detector identifies:
- **Email addresses** — john@example.com
- **US Phone numbers** — 555-123-4567, (555) 123-4567
- **Social Security Numbers** — 123-45-6789 (with validation)
- **Credit Card numbers** — 4111-1111-1111-1111 (with Luhn validation)
- **IP addresses** — 192.168.1.1 (with range validation)
- **Dates of birth** — 01/15/1990

### PII Redaction

```python
from localmod.classifiers.pii import PIIDetector

detector = PIIDetector()
detector.load()

text = "Email me at john@example.com or call 555-123-4567"
redacted, detections = detector.redact(text)
# Result: "Email me at [EMAIL] or call [PHONE]"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and version |
| `/health` | GET | Health check with loaded models |
| `/classifiers` | GET | List available classifiers |
| `/analyze` | POST | Analyze single text |
| `/analyze/batch` | POST | Analyze multiple texts |
| `/redact` | POST | Redact PII from text |

### Example: Analyze Text

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "My email is john@example.com", "classifiers": ["pii"]}'
```

Response:
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
      "metadata": {"detections": [...], "total_count": 1}
    }
  ],
  "summary": "Content flagged for: pii (medium): email",
  "processing_time_ms": 1.23
}
```

### Example: Redact PII

```bash
curl -X POST http://localhost:8000/redact \
  -H "Content-Type: application/json" \
  -d '{"text": "Contact john@example.com", "replacement": "[HIDDEN]"}'
```

Response:
```json
{
  "original_length": 24,
  "redacted_text": "Contact [HIDDEN]",
  "redactions": [{"type": "email", "start": 8, "end": 24}],
  "processing_time_ms": 0.45
}
```

## Configuration

Environment variables (prefix with `LOCALMOD_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | API server host |
| `PORT` | `8000` | API server port |
| `DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `LAZY_LOAD` | `true` | Load models on first request |
| `TOXICITY_THRESHOLD` | `0.5` | Toxicity detection threshold |
| `PII_THRESHOLD` | `0.5` | PII detection threshold |
| `PROMPT_INJECTION_THRESHOLD` | `0.5` | Prompt injection threshold |
| `SPAM_THRESHOLD` | `0.5` | Spam detection threshold |
| `NSFW_THRESHOLD` | `0.5` | NSFW detection threshold |
| `MAX_BATCH_SIZE` | `32` | Maximum batch size |
| `MAX_TEXT_LENGTH` | `10000` | Maximum text length |

## CLI Commands

```bash
# Start API server
localmod serve [--host HOST] [--port PORT] [--reload]

# Analyze text from command line
localmod analyze "Your text here" [-c classifier1 classifier2]

# Pre-download all models
localmod download

# List available classifiers
localmod list
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[all]"

# Run tests
make test

# Run fast tests (skip ML model loading)
make test-fast

# Start dev server with auto-reload
make serve

# Build Docker image
make docker-build
```

## Project Structure

```
localmod/
├── src/localmod/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration management
│   ├── schemas.py            # Pydantic models
│   ├── pipeline.py           # SafetyPipeline orchestrator
│   ├── cli.py                # Command-line interface
│   ├── models/
│   │   ├── base.py           # BaseClassifier, Severity
│   │   └── ...
│   ├── classifiers/
│   │   ├── toxicity.py       # ToxicityClassifier
│   │   ├── pii.py            # PIIDetector
│   │   ├── prompt_injection.py
│   │   ├── spam.py
│   │   └── nsfw.py
│   └── api/
│       ├── app.py            # FastAPI application
│       └── routes.py         # API endpoints
├── tests/                    # Test suite (91 tests)
├── docker/                   # Docker configurations
├── Makefile                  # Build automation
└── pyproject.toml            # Project configuration
```

## Performance

| Metric | CPU | GPU (RTX 3060+) |
|--------|-----|-----------------|
| Latency (p95) | <100ms | <30ms |
| Throughput | 20+ req/sec | 100+ req/sec |
| Memory | <2GB RAM | <4GB VRAM |

## Severity Levels

| Level | Description |
|-------|-------------|
| `none` | Content passed checks |
| `low` | Minor issues detected |
| `medium` | Moderate issues |
| `high` | Serious issues |
| `critical` | Severe issues (SSN, credit cards) |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `make test` to verify
5. Submit a pull request

## Acknowledgments

- HuggingFace Transformers
- FastAPI
- PyTorch
