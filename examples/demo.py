#!/usr/bin/env python3
"""
LocalMod Demo - Content Moderation API

This demo shows LocalMod's capabilities:
1. PII Detection (works immediately - no model download needed)
2. PII Redaction  
3. Full Pipeline (requires model download first)
4. ML-based classifiers (toxicity, spam, etc.)

Run: python examples/demo.py
"""

import sys
import json


def print_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def demo_pii_detection():
    """Demo PII detection - works without any model downloads."""
    print_header("1. PII DETECTION (No Model Required)")
    
    from localmod.classifiers.pii import PIIDetector
    
    detector = PIIDetector()
    detector.load()
    print("✅ PII detector loaded (uses regex patterns)\n")
    
    test_cases = [
        ("Email", "Contact me at john.doe@example.com for details"),
        ("Phone", "Call our office at 555-123-4567 today"),
        ("SSN", "My social security number is 123-45-6789"),
        ("Credit Card", "Pay with card 4111-1111-1111-1111"),
        ("IP Address", "Server IP is 192.168.1.100"),
        ("Safe Text", "Hello, how are you today?"),
    ]
    
    print("Testing PII Detection:")
    print("-" * 50)
    
    for label, text in test_cases:
        result = detector.predict(text)
        if result.flagged:
            status = f"🚨 FLAGGED ({result.severity.value})"
            types = ", ".join(result.categories)
            print(f"  {label:12} | {status:20} | Types: {types}")
        else:
            print(f"  {label:12} | ✅ Safe")
    
    return True


def demo_pii_redaction():
    """Demo PII redaction."""
    print_header("2. PII REDACTION")
    
    from localmod.classifiers.pii import PIIDetector
    
    detector = PIIDetector()
    detector.load()
    
    test_texts = [
        "Contact John at john@email.com or call 555-123-4567",
        "SSN: 123-45-6789, CC: 4111-1111-1111-1111",
        "Server at 192.168.1.1, admin@company.com",
    ]
    
    print("Redaction Examples:")
    print("-" * 50)
    
    for text in test_texts:
        redacted, detections = detector.redact(text)
        print(f"\n  Original:  {text}")
        print(f"  Redacted:  {redacted}")
        print(f"  Found:     {len(detections)} PII item(s)")
    
    return True


def demo_pipeline_pii_only():
    """Demo the full pipeline with PII only (no model download needed)."""
    print_header("3. SAFETY PIPELINE (PII Only)")
    
    from localmod import SafetyPipeline
    
    # Initialize pipeline with only PII classifier
    pipeline = SafetyPipeline(classifiers=["pii"])
    print("✅ Pipeline initialized with PII classifier\n")
    
    test_texts = [
        "Hello, nice to meet you!",
        "My email is test@example.com",
        "SSN 123-45-6789, phone 555-1234",
    ]
    
    print("Pipeline Analysis:")
    print("-" * 50)
    
    for text in test_texts:
        report = pipeline.analyze(text, include_explanation=True)
        
        status = "🚨 FLAGGED" if report.flagged else "✅ Safe"
        print(f"\n  Text: \"{text[:40]}...\"" if len(text) > 40 else f"\n  Text: \"{text}\"")
        print(f"  Status: {status}")
        print(f"  Severity: {report.severity.value}")
        print(f"  Time: {report.processing_time_ms:.2f}ms")
        print(f"  Summary: {report.summary}")
    
    return True


def demo_batch_analysis():
    """Demo batch analysis."""
    print_header("4. BATCH ANALYSIS")
    
    from localmod import SafetyPipeline
    
    pipeline = SafetyPipeline(classifiers=["pii"])
    
    texts = [
        "Normal message here",
        "Email: user@domain.com", 
        "Call 555-123-4567",
        "Just a greeting!",
        "SSN is 123-45-6789",
    ]
    
    print(f"Analyzing {len(texts)} texts in batch...")
    print("-" * 50)
    
    reports = pipeline.analyze_batch(texts)
    
    flagged_count = sum(1 for r in reports if r.flagged)
    
    for i, (text, report) in enumerate(zip(texts, reports)):
        status = "🚨" if report.flagged else "✅"
        print(f"  {i+1}. {status} \"{text[:30]}...\"" if len(text) > 30 else f"  {i+1}. {status} \"{text}\"")
    
    print(f"\n  Total: {flagged_count}/{len(texts)} flagged")
    
    return True


def demo_ml_classifiers():
    """Demo ML-based classifiers (requires model download)."""
    print_header("5. ML CLASSIFIERS (Requires Model Download)")
    
    try:
        from localmod.classifiers import ToxicityClassifier
        
        print("Loading toxicity classifier...")
        classifier = ToxicityClassifier(device="cpu")
        classifier.load()
        print("✅ Toxicity classifier loaded!\n")
        
        test_texts = [
            ("Toxic", "You're a complete idiot and I hate you!"),
            ("Threatening", "I'm going to find you and hurt you"),
            ("Safe", "I hope you have a wonderful day!"),
            ("Neutral", "The weather is nice today"),
        ]
        
        print("Toxicity Detection:")
        print("-" * 50)
        
        for label, text in test_texts:
            result = classifier.predict(text)
            if result.flagged:
                print(f"  {label:12} | 🚨 TOXIC ({result.confidence:.1%}) | {result.severity.value}")
            else:
                print(f"  {label:12} | ✅ Safe ({1-result.confidence:.1%} safe)")
        
        return True
        
    except Exception as e:
        print(f"⚠️  ML classifiers not available: {type(e).__name__}")
        print(f"   {str(e)[:100]}...")
        print("\n📥 To enable ML classifiers, run:")
        print("   python scripts/download_models.py")
        print("\n   Or download specific models:")
        print("   python scripts/download_models.py --classifier toxicity")
        return False


def demo_api_example():
    """Show API usage example."""
    print_header("6. API USAGE EXAMPLE")
    
    print("To start the API server:")
    print("-" * 50)
    print("  python -m localmod.cli serve --port 8000")
    print()
    print("Then make requests:")
    print("-" * 50)
    print('''
  # Analyze text
  curl -X POST http://localhost:8000/analyze \\
    -H "Content-Type: application/json" \\
    -d '{"text": "My email is test@example.com", "classifiers": ["pii"]}'

  # Redact PII
  curl -X POST http://localhost:8000/redact \\
    -H "Content-Type: application/json" \\
    -d '{"text": "Call me at 555-123-4567"}'

  # Health check
  curl http://localhost:8000/health
''')
    return True


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("  🛡️  LocalMod Demo - Content Moderation API")
    print("="*60)
    print("\nLocalMod provides 5 safety classifiers:")
    print("  • Toxicity    - Hate speech, harassment (ML)")
    print("  • PII         - Emails, phones, SSNs (Regex)")
    print("  • Injection   - Prompt injection attacks (ML)")
    print("  • Spam        - Promotional content (ML)")
    print("  • NSFW        - Adult content (ML)")
    print("\n⚡ PII detection works immediately - no model download needed!")
    
    results = {}
    
    # Run demos
    results["PII Detection"] = demo_pii_detection()
    results["PII Redaction"] = demo_pii_redaction()
    results["Pipeline"] = demo_pipeline_pii_only()
    results["Batch Analysis"] = demo_batch_analysis()
    results["ML Classifiers"] = demo_ml_classifiers()
    demo_api_example()
    
    # Summary
    print_header("DEMO SUMMARY")
    
    for name, success in results.items():
        status = "✅ Passed" if success else "⚠️  Skipped (needs model download)"
        print(f"  {name:20} | {status}")
    
    print("\n" + "="*60)
    print("  Demo Complete!")
    print("="*60)
    
    if not results.get("ML Classifiers"):
        print("\n💡 To enable all features, download the ML models:")
        print("   python scripts/download_models.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
