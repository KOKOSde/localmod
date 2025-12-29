# Changelog

All notable changes to LocalMod will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-27

### Added

- **Core Framework**
  - `SafetyPipeline` - Multi-classifier orchestration
  - `SafetyReport` - Aggregated results with severity levels
  - `BaseClassifier` - Abstract base for all classifiers
  - Configuration management with environment variables

- **Classifiers**
  - `ToxicityClassifier` - Hate speech, harassment, threats detection
  - `PIIDetector` - Email, phone, SSN, credit card, IP detection
  - `PromptInjectionDetector` - LLM jailbreak and instruction override detection
  - `SpamClassifier` - Promotional content and scam detection
  - `NSFWClassifier` - Adult content detection

- **API**
  - FastAPI-based REST API
  - `POST /analyze` - Single text analysis
  - `POST /analyze/batch` - Batch analysis
  - `POST /redact` - PII redaction
  - `GET /health` - Health check
  - `GET /classifiers` - List classifiers

- **CLI**
  - `localmod serve` - Start API server
  - `localmod analyze` - Analyze text from command line
  - `localmod download` - Pre-download models
  - `localmod list` - List available classifiers

- **Docker**
  - CPU Dockerfile with multi-stage build
  - GPU Dockerfile with CUDA support
  - Docker Compose configuration

- **Testing**
  - 91 unit and integration tests
  - 74% code coverage

### Technical Details

- Python 3.6+ support
- PyTorch and Transformers integration
- Pydantic for request/response validation
- Configurable thresholds per classifier
- Lazy model loading option


