#!/usr/bin/env python
"""Demo script to test LocalMod functionality."""

import json
import sys
import time

# Check if running with models downloaded
def main():
    print("=" * 60)
    print("LocalMod Demo - Content Moderation API")
    print("=" * 60)
    
    from localmod import SafetyPipeline
    from localmod.classifiers import PIIDetector
    
    # Initialize pipeline (PII only for quick demo - no model downloads needed)
    print("\n[1] Initializing PII detector (no ML model needed)...")
    pii = PIIDetector()
    pii.load()
    print("    ✓ PII detector ready")
    
    # Test PII Detection
    print("\n[2] Testing PII Detection...")
    test_cases = [
        "My email is john.doe@example.com",
        "Call me at 555-123-4567",
        "SSN: 123-45-6789",
        "Credit card: 4111-1111-1111-1111",
        "Hello, how are you today?",  # No PII
    ]
    
    for text in test_cases:
        result = pii.predict(text)
        status = "🚨 FLAGGED" if result.flagged else "✅ Safe"
        print(f"    {status}: \"{text[:40]}...\"")
        if result.flagged:
            print(f"       Types: {result.categories}")
    
    # Test PII Redaction
    print("\n[3] Testing PII Redaction...")
    text = "Contact John at john@email.com or 555-123-4567"
    redacted, _ = pii.redact(text)
    print(f"    Original:  {text}")
    print(f"    Redacted:  {redacted}")
    
    # Full pipeline test (requires model downloads)
    print("\n[4] Testing Full Pipeline (requires models)...")
    try:
        pipeline = SafetyPipeline(classifiers=["pii"])  # Just PII for quick test
        report = pipeline.analyze(
            "My email is test@example.com and I hate everyone!",
            include_explanation=True
        )
        print(f"    Flagged: {report.flagged}")
        print(f"    Severity: {report.severity.value}")
        print(f"    Summary: {report.summary}")
        print(f"    Time: {report.processing_time_ms:.2f}ms")
    except Exception as e:
        print(f"    ⚠ Full pipeline test skipped: {e}")
    
    # Test with ML models if available
    print("\n[5] Testing ML-based classifiers (if models are downloaded)...")
    try:
        from localmod.classifiers import ToxicityClassifier
        toxicity = ToxicityClassifier(device="cpu")
        toxicity.load()
        
        toxic_texts = [
            "You're a complete idiot!",
            "I hope you have a wonderful day!",
            "Die in a fire you moron",
        ]
        
        print("    Toxicity classifier loaded!")
        for text in toxic_texts:
            result = toxicity.predict(text)
            status = "🚨 TOXIC" if result.flagged else "✅ Safe"
            print(f"    {status} ({result.confidence:.2%}): \"{text[:40]}\"")
            
    except Exception as e:
        print(f"    ⚠ ML classifiers not available: {type(e).__name__}")
        print("    Run 'python -m localmod.cli download' to download models")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

