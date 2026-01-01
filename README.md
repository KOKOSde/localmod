# LocalMod

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-109%20passed-brightgreen.svg)]()
[![Toxicity Accuracy](https://img.shields.io/badge/toxicity-0.75%20balanced%20acc-blue.svg)]()

**Fully offline content moderation API** — Self-hosted content safety for teams who can't afford to send data to the cloud.

<p align="center">
  <img src="docs/architecture.svg" alt="LocalMod Architecture" width="800"/>
</p>

<p align="center">
  <img src="docs/examples.svg" alt="LocalMod Examples" width="900"/>
</p>

---

## 📊 Benchmark Results

LocalMod was evaluated on established public benchmarks:

| Benchmark | Task | LocalMod | OpenAI | Azure | Amazon | Perspective |
|-----------|------|----------|--------|-------|--------|-------------|
| **CHI 2025** ¹ | Toxicity | **0.75** | 0.83 | 0.81 | 0.74 | 0.62 |
| **SMS Spam** ² | Spam | **0.998** | — | — | — | — |

**Toxicity Detection**: LocalMod achieves **0.75 balanced accuracy**, matching Amazon Comprehend and outperforming Perspective API — while running 100% locally.

**Spam Detection**: LocalMod achieves **99.8% balanced accuracy** on the UCI SMS Spam Collection benchmark.

> ¹ CHI 2025 ["Lost in Moderation"](https://arxiv.org/html/2503.01623) methodology (HateXplain, Civil Comments, SBIC datasets). Commercial scores from paper.
> ² [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) - 5,574 SMS messages (Almeida et al., 2011)

---

## How It Works

```
┌─────────────┐      POST /analyze       ┌───────────────────────────────────────┐      JSON       ┌─────────────┐
│  Your App   │  ──────────────────────▶ │           LocalMod API                │ ──────────────▶ │  Response   │
│             │     "Check this text"    │  ┌─────┐ ┌────┐ ┌─────┐ ┌────┐ ┌────┐ │                 │ flagged: T/F│
└─────────────┘                          │  │Toxic│ │PII │ │Jailb│ │Spam│ │NSFW│ │                 │ severity    │
                                         │  └─────┘ └────┘ └─────┘ └────┘ └────┘ │                 │ confidence  │
                                         └───────────────────────────────────────┘                 └─────────────┘
                                                    Runs 100% Locally
```

## Why LocalMod?

| Feature | Benefit |
|---------|---------|
| 🔒 **100% Offline** | All models run locally, your data never leaves your infrastructure |
| 📊 **Benchmarked** | 0.75 balanced accuracy on CHI 2025, matching Amazon Comprehend |
| ⚡ **Fast** | <100ms on CPU, <30ms on GPU |
| 🎯 **Comprehensive** | 5 classifiers: Toxicity, PII, Prompt Injection, Spam, NSFW |
| 🐳 **Docker Ready** | Single image deployment, works anywhere |
| 💰 **Cost Effective** | No per-request API fees, one-time setup |

## Quick Start

### Installation

```bash
git clone https://github.com/KOKOSde/localmod.git
cd localmod
pip install -e .
```

### Run the Demo (Works Immediately!)

```bash
python examples/demo.py
```

**PII detection works instantly** — no model download needed!

### Download ML Models

For toxicity, spam, NSFW, and prompt injection detection:

```bash
python scripts/download_models.py
```

This downloads 7 models (~3GB total):
- **Toxicity ensemble** (4 models for 0.75 accuracy)
- Prompt injection, spam, and NSFW detectors

### Verify Models (Optional)

```bash
localmod verify-models --offline
```

### Use in Python

```python
from localmod import SafetyPipeline

pipeline = SafetyPipeline()

# Analyze text
report = pipeline.analyze("Hello, how are you?")
print(f"Flagged: {report.flagged}")  # False

# Detect PII
report = pipeline.analyze("My email is john@example.com")
print(f"Flagged: {report.flagged}")  # True
print(f"Severity: {report.severity}")  # medium
```

### Run as API Server

```bash
# Start the server
python -m localmod.cli serve --port 8000

# Test it
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "You are an idiot!", "classifiers": ["toxicity"]}'
```

### Use Docker

```bash
docker build -f docker/Dockerfile -t localmod:latest .
docker run -p 8000:8000 localmod:latest
```

---

## Offline Mode

LocalMod supports fully offline operation:

```bash
# Download models once
python scripts/download_models.py --model-dir /path/to/models

# Run offline
export LOCALMOD_MODEL_DIR=/path/to/models
export LOCALMOD_OFFLINE=1
localmod serve
```

For Docker:
```bash
docker run -p 8000:8000 \
  -v /path/to/models:/models \
  -e LOCALMOD_MODEL_DIR=/models \
  -e LOCALMOD_OFFLINE=1 \
  localmod:latest
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with loaded models |
| `/classifiers` | GET | List available classifiers |
| `/analyze` | POST | Analyze single text |
| `/analyze/batch` | POST | Analyze multiple texts |
| `/redact` | POST | Redact PII from text |

### Example API Response

```json
{
  "flagged": true,
  "results": [
    {
      "classifier": "toxicity",
      "flagged": true,
      "confidence": 0.72,
      "severity": "high",
      "categories": ["toxic"],
      "metadata": {"ensemble_models": ["toxicity", "toxicity_dehatebert", "toxicity_snlp", "toxicity_facebook"]},
      "explanation": ""
    }
  ],
  "summary": "Content flagged for: toxicity (high)",
  "processing_time_ms": 85.3
}
```

---

## Classifiers

| Classifier | Detects | Technology | Needs Download? |
|------------|---------|------------|-----------------|
| 🔒 **PII** | Emails, phones, SSNs, credit cards | Regex + Validation | ❌ **No** (works instantly) |
| 🔥 **Toxicity** | Hate speech, harassment, threats | Weighted Ensemble (4 models) | ✅ Yes |
| ⚡ **Prompt Injection** | LLM jailbreaks, instruction override | Pattern + ML (DeBERTa) | ✅ Yes |
| 📧 **Spam** | Promotional content, scams | Heuristics + ML (RoBERTa) | ✅ Yes |
| 🔞 **NSFW** | Sexual content, adult themes | ML (NSFW-classifier) | ✅ Yes |

### Toxicity Ensemble Details

The toxicity classifier uses a weighted ensemble of 4 models for optimal accuracy:

| Model | Weight | Purpose |
|-------|--------|---------|
| [unitary/toxic-bert](https://huggingface.co/unitary/toxic-bert) | 50% | Multi-label toxicity (Jigsaw) |
| [Hate-speech-CNERG/dehatebert-mono-english](https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english) | 20% | Hate speech detection |
| [s-nlp/roberta_toxicity_classifier](https://huggingface.co/s-nlp/roberta_toxicity_classifier) | 15% | Toxicity classification |
| [facebook/roberta-hate-speech-dynabench-r4-target](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) | 15% | Adversarially robust |

> 💡 **PII detection uses regex patterns with validation (Luhn for credit cards, format checks for SSNs) — no ML models required!**

### PII Detection Details

Automatically detects and can redact:
- **Email addresses** — `john@example.com` → `[EMAIL]`
- **Phone numbers** — `555-123-4567` → `[PHONE]`
- **SSNs** — `123-45-6789` → `[SSN]` (with validation)
- **Credit Cards** — `4111-1111-1111-1111` → `[CREDIT_CARD]` (Luhn validated)
- **IP addresses** — `192.168.1.1` → `[IP_ADDRESS]`

```python
from localmod.classifiers.pii import PIIDetector

detector = PIIDetector()
detector.load()

text = "Email me at john@example.com or call 555-123-4567"
redacted, _ = detector.redact(text)
# Result: "Email me at [EMAIL] or call [PHONE]"
```

---

## Configuration

Set via environment variables (prefix with `LOCALMOD_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `~/.cache/localmod/models` | Directory for downloaded models |
| `OFFLINE` | `false` | Force offline mode (local models only) |
| `DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `LAZY_LOAD` | `true` | Load models on first request |
| `TOXICITY_THRESHOLD` | `0.17` | Toxicity detection threshold |
| `MAX_TEXT_LENGTH` | `10000` | Maximum text length |

---

## Performance

| Metric | CPU | GPU |
|--------|-----|-----|
| Latency (p95) | <100ms | <30ms |
| Throughput | 20+ req/sec | 100+ req/sec |
| Memory | <4GB RAM | <6GB VRAM |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run fast tests (PII only, no model downloads)
pytest tests/test_classifiers/test_pii.py -v
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments / Models

LocalMod uses pre-trained models from [HuggingFace](https://huggingface.co/):

**Toxicity Ensemble (4 models):**
- [`unitary/toxic-bert`](https://huggingface.co/unitary/toxic-bert) — Multi-label toxicity (50% weight)
- [`Hate-speech-CNERG/dehatebert-mono-english`](https://huggingface.co/Hate-speech-CNERG/dehatebert-mono-english) — Hate speech (20% weight)
- [`s-nlp/roberta_toxicity_classifier`](https://huggingface.co/s-nlp/roberta_toxicity_classifier) — Toxicity (15% weight)
- [`facebook/roberta-hate-speech-dynabench-r4-target`](https://huggingface.co/facebook/roberta-hate-speech-dynabench-r4-target) — Adversarial (15% weight)

**Other Classifiers:**
- [`deepset/deberta-v3-base-injection`](https://huggingface.co/deepset/deberta-v3-base-injection) — Prompt Injection
- [`mshenoda/roberta-spam`](https://huggingface.co/mshenoda/roberta-spam) — Spam
- [`michellejieli/NSFW_text_classifier`](https://huggingface.co/michellejieli/NSFW_text_classifier) — NSFW

**Benchmark Reference:**
- CHI 2025 "Lost in Moderation" paper methodology
- Datasets: HateXplain, Civil Comments, SBIC
