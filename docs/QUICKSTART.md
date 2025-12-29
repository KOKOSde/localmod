# LocalMod Quick Start Guide

Get LocalMod running in 5 minutes.

## Prerequisites

- Python 3.8+ 
- pip
- ~2GB disk space for ML models

## Step 1: Install

```bash
git clone https://github.com/KOKOSde/localmod.git
cd localmod
pip install -e .
```

## Step 2: Download Models

```bash
# Download all ML models (one-time, ~500MB)
python scripts/download_models.py

# Or specify custom directory
python scripts/download_models.py --model-dir /path/to/models
```

## Step 3: Verify Offline Readiness

```bash
localmod verify-models --offline
```

Expected output:
```
🔍 LocalMod Model Verification
📁 Model directory: ~/.cache/localmod/models
🔒 Offline mode: Yes

📦 toxicity
   Downloaded: ✅ Yes
   Loads: ✅ Yes
   Flags 'threat': ✅ Yes (91%)
   Passes 'greeting': ✅ Yes (0%)

📦 prompt_injection
   Downloaded: ✅ Yes
   ...

✅ All models verified successfully!
```

## Step 4: Use LocalMod

### Python API

```python
from localmod import SafetyPipeline

pipeline = SafetyPipeline()

# Analyze text
report = pipeline.analyze("Hello, how are you?")
print(f"Flagged: {report.flagged}")  # False

# PII detection (no ML model needed)
report = pipeline.analyze("My email is john@example.com", classifiers=["pii"])
print(f"Flagged: {report.flagged}")  # True
```

### REST API

```bash
# Start server
localmod serve --port 8000

# Test
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "My email is test@example.com", "classifiers": ["pii"]}'
```

## Step 5: Run Offline

```bash
export LOCALMOD_MODEL_DIR=~/.cache/localmod/models
export LOCALMOD_OFFLINE=1
localmod serve
```

---

## What Works Without Model Downloads?

| Classifier | Requires Download? |
|------------|-------------------|
| PII | ❌ No (regex-based) |
| Toxicity | ✅ Yes |
| Prompt Injection | ✅ Yes |
| Spam | ✅ Yes |
| NSFW | ✅ Yes |

---

## Next Steps

- [API Documentation](API.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Examples](../examples/)
