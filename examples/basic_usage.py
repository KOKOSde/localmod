#!/usr/bin/env python3
"""
Basic LocalMod Usage Examples

Copy-paste these examples to get started quickly.
PII detection works immediately - no model download needed!
"""

# =============================================================================
# EXAMPLE 1: Quick PII Check (No Model Download Needed)
# =============================================================================

from localmod import SafetyPipeline

# Create pipeline with PII only (works instantly)
pipeline = SafetyPipeline(classifiers=["pii"])

# Check some text
text = "Contact me at john@example.com"
report = pipeline.analyze(text)

print(f"Text: {text}")
print(f"Flagged: {report.flagged}")          # True
print(f"Severity: {report.severity.value}")  # medium
print(f"Summary: {report.summary}")
print()

# =============================================================================
# EXAMPLE 2: PII Redaction
# =============================================================================

from localmod.classifiers.pii import PIIDetector

detector = PIIDetector()
detector.load()

text = "Email: john@example.com, Phone: 555-123-4567, SSN: 123-45-6789"
redacted, detections = detector.redact(text)

print(f"Original: {text}")
print(f"Redacted: {redacted}")
print(f"Found {len(detections)} PII items")
print()

# =============================================================================
# EXAMPLE 3: Batch Processing
# =============================================================================

texts = [
    "Hello, nice weather today!",
    "My email is user@domain.com",
    "Call me at 555-123-4567",
    "Credit card: 4111-1111-1111-1111",
]

reports = pipeline.analyze_batch(texts)

print("Batch Results:")
for text, report in zip(texts, reports):
    status = "🚨 FLAGGED" if report.flagged else "✅ Safe"
    print(f"  {status}: {text[:40]}")
print()

# =============================================================================
# EXAMPLE 4: Custom Thresholds
# =============================================================================

# You can customize detection thresholds
pipeline_strict = SafetyPipeline(
    classifiers=["pii"],
    thresholds={"pii": 0.3}  # More sensitive
)

# =============================================================================
# EXAMPLE 5: Full Pipeline (Requires Model Download)
# =============================================================================

# To use all classifiers, first download models:
#   python scripts/download_models.py

# Then:
# full_pipeline = SafetyPipeline()  # Uses all classifiers
# report = full_pipeline.analyze("You're an idiot!")
# print(f"Toxic: {report.flagged}")

print("✅ Examples complete!")
print()
print("To enable toxicity/spam/NSFW detection, run:")
print("  python scripts/download_models.py")

