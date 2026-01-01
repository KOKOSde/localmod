#!/usr/bin/env python3
"""
Download all ML models required by LocalMod to a local directory.

This script downloads models from HuggingFace and saves them locally.
Run this once before using LocalMod in offline mode.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model-dir /path/to/models
    python scripts/download_models.py --classifier toxicity
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Model registry - maps classifier names to their HuggingFace model IDs
# Toxicity uses a weighted ensemble of 4 models for 0.75 balanced accuracy
MODELS: Dict[str, str] = {
    # Toxicity ensemble (weighted: 50% unitary, 20% dehatebert, 15% snlp, 15% facebook)
    "toxicity": "unitary/toxic-bert",
    "toxicity_dehatebert": "Hate-speech-CNERG/dehatebert-mono-english",
    "toxicity_snlp": "s-nlp/roberta_toxicity_classifier",
    "toxicity_facebook": "facebook/roberta-hate-speech-dynabench-r4-target",
    # Other text classifiers
    "prompt_injection": "deepset/deberta-v3-base-injection",
    "spam": "mshenoda/roberta-spam",
    "nsfw": "michellejieli/NSFW_text_classifier",
    # Image classifiers
    "nsfw_image": "Falconsai/nsfw_image_detection",
}

# Download order (stable, user-facing) - text models first, then image
DOWNLOAD_ORDER: List[str] = [
    "toxicity", "toxicity_dehatebert", "toxicity_snlp", "toxicity_facebook",
    "prompt_injection", "spam", "nsfw",
    "nsfw_image",
]

# Model descriptions for display
MODEL_DESCRIPTIONS: Dict[str, str] = {
    "toxicity": "Toxicity (Unitary Toxic-BERT) - 50% ensemble weight",
    "toxicity_dehatebert": "Toxicity (DeHateBERT) - 20% ensemble weight",
    "toxicity_snlp": "Toxicity (s-nlp RoBERTa) - 15% ensemble weight",
    "toxicity_facebook": "Toxicity (Facebook Dynabench) - 15% ensemble weight",
    "prompt_injection": "Prompt Injection Detection",
    "spam": "Spam Detection",
    "nsfw": "NSFW Text Content Detection",
    "nsfw_image": "NSFW Image Detection (ViT)",
}


def get_default_model_dir() -> str:
    """Get default model directory."""
    env_dir = os.environ.get("LOCALMOD_MODEL_DIR")
    if env_dir:
        return os.path.expanduser(env_dir)
    return os.path.expanduser("~/.cache/localmod/models")


# Image classifier names (use different download logic)
IMAGE_CLASSIFIERS = {"nsfw_image"}


def download_model(name: str, model_id: str, target_dir: str) -> bool:
    """
    Download a model from HuggingFace and save to local directory.
    
    Args:
        name: Classifier name
        model_id: HuggingFace model ID
        target_dir: Directory to save the model
        
    Returns:
        True if successful, False otherwise
    """
    desc = MODEL_DESCRIPTIONS.get(name, name)
    print(f"\n{'='*60}")
    print(f"📥 Downloading: {desc}")
    print(f"   From: {model_id}")
    print(f"   To:   {target_dir}")
    print(f"{'='*60}")
    
    try:
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        if name in IMAGE_CLASSIFIERS:
            # Image classifier - use AutoImageProcessor
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            
            print("   [1/4] Loading image processor...")
            processor = AutoImageProcessor.from_pretrained(model_id)
            
            print("   [2/4] Loading model...")
            model = AutoModelForImageClassification.from_pretrained(model_id)
            
            print("   [3/4] Saving image processor...")
            processor.save_pretrained(target_dir)
            
            print("   [4/4] Saving model...")
            model.save_pretrained(target_dir)
        else:
            # Text classifier - use AutoTokenizer
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            print("   [1/4] Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            print("   [2/4] Loading model...")
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            
            print("   [3/4] Saving tokenizer...")
            tokenizer.save_pretrained(target_dir)
            
            print("   [4/4] Saving model...")
            model.save_pretrained(target_dir)
        
        print(f"   ✅ {name} downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to download {name}: {e}")
        return False


def write_manifest(model_dir: str, downloaded: Dict[str, str]) -> None:
    """Write manifest.json with download metadata."""
    import torch
    import transformers
    
    manifest = {
        "downloaded_at": datetime.now().isoformat(),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "ensemble_info": {
            "toxicity_models": ["toxicity", "toxicity_dehatebert", "toxicity_snlp", "toxicity_facebook"],
            "toxicity_weights": {
                "toxicity": 0.50,
                "toxicity_dehatebert": 0.20,
                "toxicity_snlp": 0.15,
                "toxicity_facebook": 0.15,
            },
            "benchmark_score": "0.75 balanced accuracy (CHI 2025)",
        },
        "models": {
            name: {
                "hf_model_id": model_id,
                "local_path": os.path.join(model_dir, name),
                "description": MODEL_DESCRIPTIONS.get(name, name),
            }
            for name, model_id in downloaded.items()
        }
    }
    
    manifest_path = os.path.join(model_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📄 Manifest written to: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download LocalMod ML models from HuggingFace"
    )
    parser.add_argument(
        "--classifier", "-c",
        choices=list(MODELS.keys()) + ["all", "pii", "toxicity_all"],
        default="all",
        help="Which classifier's model to download (default: all). Use 'toxicity_all' for all toxicity ensemble models."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=f"Directory to save models (default: $LOCALMOD_MODEL_DIR or ~/.cache/localmod/models)"
    )
    args = parser.parse_args()
    
    # Determine model directory
    model_dir = args.model_dir or get_default_model_dir()
    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    
    print("\n" + "="*60)
    print("🚀 LocalMod Model Downloader")
    print("="*60)
    print(f"\n📁 Model directory: {model_dir}")
    
    # Handle PII specially - no model needed
    if args.classifier == "pii":
        print("\n✅ PII detector uses regex patterns - no model download needed!")
        return 0
    
    # Determine which models to download (in stable order)
    if args.classifier == "all":
        to_download = [(name, MODELS[name]) for name in DOWNLOAD_ORDER]
    elif args.classifier == "toxicity_all":
        toxicity_models = ["toxicity", "toxicity_dehatebert", "toxicity_snlp", "toxicity_facebook"]
        to_download = [(name, MODELS[name]) for name in toxicity_models]
    elif args.classifier.startswith("toxicity"):
        to_download = [(args.classifier, MODELS[args.classifier])]
    else:
        to_download = [(args.classifier, MODELS[args.classifier])]
    
    print(f"\n📋 Models to download:")
    for name, model_id in to_download:
        desc = MODEL_DESCRIPTIONS.get(name, name)
        print(f"   • {desc}")
        print(f"     {model_id}")
    print(f"\n   Total: {len(to_download)} model(s)")
    
    # Check for transformers
    try:
        import transformers
        import torch
        print(f"\n🔧 transformers: {transformers.__version__}")
        print(f"🔧 torch: {torch.__version__}")
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("   Run: pip install transformers torch")
        return 1
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Download each model
    results: Dict[str, bool] = {}
    downloaded: Dict[str, str] = {}
    
    for name, model_id in to_download:
        target_dir = os.path.join(model_dir, name)
        success = download_model(name, model_id, target_dir)
        results[name] = success
        if success:
            downloaded[name] = model_id
    
    # Write manifest
    if downloaded:
        write_manifest(model_dir, downloaded)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Download Summary")
    print("="*60)
    
    success_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - success_count
    
    for name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        desc = MODEL_DESCRIPTIONS.get(name, name)
        print(f"   {desc}: {status}")
    
    print(f"\n   Total: {success_count}/{len(results)} succeeded")
    
    if fail_count > 0:
        print("\n⚠️  Some downloads failed. Check your internet connection and try again.")
        return 1
    
    print("\n" + "="*60)
    print("✅ All models downloaded!")
    print("="*60)
    print(f"\n📁 Models saved to: {model_dir}")
    print("\n📊 Toxicity Ensemble Info:")
    print("   4 models combined achieve 0.75 balanced accuracy")
    print("   (matches Amazon Comprehend, beats Perspective API)")
    print("\n🔒 To run offline:")
    print(f"   export LOCALMOD_MODEL_DIR={model_dir}")
    print("   export LOCALMOD_OFFLINE=1")
    print("\n🔍 To verify:")
    print("   localmod verify-models --offline")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
