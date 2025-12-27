#!/usr/bin/env python3
"""
Download all models required by LocalMod.

This script downloads models from HuggingFace and caches them locally.
Run this once before using LocalMod in offline mode.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --classifier toxicity
    python scripts/download_models.py --classifier pii  # No download needed (regex-based)
"""

import argparse
import sys
import os

# Model registry - maps classifier names to their HuggingFace model IDs
# Updated to use better-performing models
MODELS = {
    "toxicity": "unitary/toxic-bert",  # Multi-label, better accuracy
    "prompt_injection": "deepset/deberta-v3-base-injection",  # pytorch format
    "spam": "mshenoda/roberta-spam",  # Better general spam detection
    "nsfw": "michellejieli/NSFW_text_classifier",  # Has some issues with "puppies"
    # "pii" doesn't need a model - it uses regex patterns
}


def download_model(name: str, model_id: str) -> bool:
    """Download a single model from HuggingFace."""
    print(f"\n{'='*60}")
    print(f"📥 Downloading: {name}")
    print(f"   Model: {model_id}")
    print(f"{'='*60}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print("   Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        
        print(f"   ✅ {name} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to download {name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download LocalMod ML models from HuggingFace"
    )
    parser.add_argument(
        "--classifier", "-c",
        choices=list(MODELS.keys()) + ["all", "pii"],
        default="all",
        help="Which classifier's model to download (default: all)"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom cache directory for models"
    )
    args = parser.parse_args()
    
    # Set cache directory if specified
    if args.cache_dir:
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
        os.environ["HF_HOME"] = args.cache_dir
        print(f"📁 Using cache directory: {args.cache_dir}")
    
    print("\n" + "="*60)
    print("🚀 LocalMod Model Downloader")
    print("="*60)
    
    # Handle PII specially - no model needed
    if args.classifier == "pii":
        print("\n✅ PII detector uses regex patterns - no model download needed!")
        return 0
    
    # Determine which models to download
    if args.classifier == "all":
        models_to_download = MODELS
    else:
        models_to_download = {args.classifier: MODELS[args.classifier]}
    
    print(f"\n📋 Models to download: {', '.join(models_to_download.keys())}")
    print(f"   Total: {len(models_to_download)} model(s)")
    
    # Check for transformers
    try:
        import transformers
        print(f"\n🔧 Using transformers version: {transformers.__version__}")
    except ImportError:
        print("\n❌ Error: transformers not installed!")
        print("   Run: pip install transformers torch")
        return 1
    
    # Download each model
    results = {}
    for name, model_id in models_to_download.items():
        results[name] = download_model(name, model_id)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Download Summary")
    print("="*60)
    
    success_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - success_count
    
    for name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {name}: {status}")
    
    print(f"\n   Total: {success_count}/{len(results)} succeeded")
    
    if fail_count > 0:
        print("\n⚠️  Some downloads failed. Check your internet connection and try again.")
        print("   You can also try downloading specific models:")
        print("   python scripts/download_models.py --classifier toxicity")
        return 1
    
    print("\n✅ All models downloaded! LocalMod is ready for offline use.")
    print("\n📖 Quick Start:")
    print("   from localmod import SafetyPipeline")
    print("   pipeline = SafetyPipeline()")
    print('   result = pipeline.analyze("Your text here")')
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

