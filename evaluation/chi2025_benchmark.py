#!/usr/bin/env python3
"""
CHI 2025 Benchmark for LocalMod Toxicity Detection.

This script evaluates LocalMod's toxicity detection against the CHI 2025
"Lost in Moderation" benchmark methodology using:
- HateXplain dataset
- Civil Comments dataset  
- SBIC (Social Bias Inference Corpus)

Results are compared with published scores from commercial APIs.

Usage:
    python evaluation/chi2025_benchmark.py
    python evaluation/chi2025_benchmark.py --samples 500  # Quick test
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Published commercial API scores from CHI 2025 paper
# Reference: Hartmann et al., "Lost in Moderation", CHI 2025
# https://arxiv.org/html/2503.01623
PUBLISHED_SCORES = {
    "OpenAI Content Moderation API": 0.83,
    "Microsoft Azure Content Moderation": 0.81,
    "Amazon Comprehend": 0.74,
    "Perspective API": 0.62,
    "Google Natural Language API": 0.59,
}

# LocalMod toxicity ensemble configuration
ENSEMBLE_CONFIG = {
    "toxicity": {"weight": 0.50, "type": "multilabel", "path": "~/.cache/localmod/models/toxicity"},
    "toxicity_dehatebert": {"weight": 0.20, "type": "binary", "path": "~/.cache/localmod/models/toxicity_dehatebert"},
    "toxicity_snlp": {"weight": 0.15, "type": "binary", "path": "~/.cache/localmod/models/toxicity_snlp"},
    "toxicity_facebook": {"weight": 0.15, "type": "binary", "path": "~/.cache/localmod/models/toxicity_facebook"},
}

# Alternative paths if models are saved with different names
ALTERNATIVE_PATHS = {
    "toxicity_dehatebert": "~/.cache/localmod/models/dehatebert",
    "toxicity_snlp": "~/.cache/localmod/models/snlp_toxicity",
    "toxicity_facebook": "~/.cache/localmod/models/fb_hate_speech",
}


def load_datasets(max_samples: int = None) -> Dict[str, List[Tuple[str, int]]]:
    """Load benchmark datasets."""
    print("\nLoading datasets...")
    datasets = {}
    
    # HateXplain
    from collections import Counter
    ds = load_dataset("hatexplain", split="test")
    hatexplain = []
    for item in ds:
        text = " ".join(item["post_tokens"])
        labels = item["annotators"]["label"]
        majority_label = Counter(labels).most_common(1)[0][0]
        hatexplain.append((text, 1 if majority_label > 0 else 0))
        if max_samples and len(hatexplain) >= max_samples:
            break
    datasets["HateXplain"] = hatexplain
    print(f"  HateXplain: {len(hatexplain)} samples")
    
    # Civil Comments
    sample_size = max_samples or 2000
    ds = load_dataset("google/civil_comments", split=f"test[:{sample_size}]")
    civil = [(item["text"][:512], 1 if item["toxicity"] >= 0.5 else 0) for item in ds]
    datasets["Civil Comments"] = civil
    print(f"  Civil Comments: {len(civil)} samples")
    
    # SBIC
    ds = load_dataset("social_bias_frames", split=f"test[:{sample_size}]")
    sbic = []
    for item in ds:
        try:
            score = float(item["offensiveYN"]) if item["offensiveYN"] else 0.0
        except:
            score = 0.0
        sbic.append((item["post"][:512], 1 if score >= 0.5 else 0))
    datasets["SBIC"] = sbic
    print(f"  SBIC: {len(sbic)} samples")
    
    return datasets


def load_ensemble_models(device: str = "cpu") -> Dict[str, Dict]:
    """Load all toxicity ensemble models."""
    print("\nLoading ensemble models...")
    models = {}
    
    for name, config in ENSEMBLE_CONFIG.items():
        path = os.path.expanduser(config["path"])
        
        # Try alternative path if primary doesn't exist
        if not os.path.exists(path) and name in ALTERNATIVE_PATHS:
            path = os.path.expanduser(ALTERNATIVE_PATHS[name])
        
        if not os.path.exists(path):
            print(f"  ⚠️  {name}: Not found at {path}")
            continue
        
        try:
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to(device)
            model.eval()
            models[name] = {
                "tokenizer": tok,
                "model": model,
                "type": config["type"],
                "weight": config["weight"],
            }
            print(f"  ✅ {name}: Loaded ({config['weight']*100:.0f}% weight)")
        except Exception as e:
            print(f"  ❌ {name}: Failed - {e}")
    
    return models


def get_ensemble_predictions(
    models: Dict[str, Dict],
    texts: List[str],
    threshold: float = 0.17,
    batch_size: int = 32,
    device: str = "cpu",
) -> List[int]:
    """Get predictions using weighted ensemble."""
    n = len(texts)
    weighted_probs = np.zeros(n)
    total_weight = 0.0
    
    for name, m in models.items():
        all_probs = []
        
        for i in range(0, n, batch_size):
            batch = texts[i:i+batch_size]
            inputs = m["tokenizer"](
                batch, return_tensors="pt", truncation=True, max_length=256, padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                out = m["model"](**inputs)
            
            if m["type"] == "multilabel":
                probs = torch.sigmoid(out.logits).max(dim=-1).values
            else:
                probs = torch.softmax(out.logits, dim=-1)[:, 1]
            
            all_probs.extend(probs.cpu().tolist())
        
        weighted_probs += np.array(all_probs) * m["weight"]
        total_weight += m["weight"]
    
    # Normalize if not all models loaded
    if total_weight > 0 and total_weight < 1.0:
        weighted_probs /= total_weight
    
    return (weighted_probs >= threshold).astype(int).tolist()


def run_benchmark(max_samples: int = None, device: str = "cpu") -> Dict:
    """Run the full CHI 2025 benchmark."""
    print("=" * 70)
    print("CHI 2025 Benchmark - LocalMod Toxicity Detection")
    print("=" * 70)
    
    # Load data
    datasets = load_datasets(max_samples)
    
    # Load models
    models = load_ensemble_models(device)
    
    if not models:
        print("\n❌ No models loaded. Run 'python scripts/download_models.py' first.")
        return {}
    
    # Evaluate on each dataset
    results = {}
    
    for ds_name, data in datasets.items():
        print(f"\nEvaluating on {ds_name}...")
        start = time.time()
        
        texts = [t for t, _ in data]
        labels = [l for _, l in data]
        
        preds = get_ensemble_predictions(models, texts, device=device)
        
        bal_acc = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        
        elapsed = time.time() - start
        
        results[ds_name] = {
            "balanced_accuracy": bal_acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "samples": len(data),
            "time_seconds": elapsed,
        }
        
        print(f"  Balanced Accuracy: {bal_acc:.3f}")
        print(f"  F1 Score: {f1:.3f}")
        print(f"  Time: {elapsed:.1f}s")
    
    # Calculate average
    avg_bal_acc = np.mean([r["balanced_accuracy"] for r in results.values()])
    avg_f1 = np.mean([r["f1_score"] for r in results.values()])
    
    results["average"] = {
        "balanced_accuracy": avg_bal_acc,
        "f1_score": avg_f1,
    }
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH COMMERCIAL APIs")
    print("=" * 70)
    
    all_scores = {**PUBLISHED_SCORES, "LocalMod (Ensemble)": avg_bal_acc}
    for name, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
        marker = "⭐" if "LocalMod" in name else ""
        print(f"{name:35} {score:.2f} {marker}")
    
    print(f"\n🏆 LocalMod achieved {avg_bal_acc:.3f} balanced accuracy")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "models_loaded": list(models.keys()),
        "threshold": 0.17,
        "datasets": results,
        "comparison": all_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Run CHI 2025 benchmark")
    parser.add_argument("--samples", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()
    
    results = run_benchmark(max_samples=args.samples, device=args.device)
    
    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
